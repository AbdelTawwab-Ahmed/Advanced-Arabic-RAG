import os
import re
from dataclasses import dataclass
from typing import List, Optional

MAX_TOKENS_PER_CHUNK = 400
AVG_TOKENS_PER_WORD = 1.5
TEXT_TITLE_MAX_WORDS = 12

BOILERPLATE_TOKENS = {
    "رقم النسخة", "تاريخ الإصدار", "تاريخ آخر مراجعة",
    "اسم الاجراء", "بنك الإسكان", "Housing Bank",
}


@dataclass
class Chunk:
    text: str
    source_pdf: str
    page_number: int
    chunk_index: int

    def to_dict(self) -> dict:
        return {
            "id": f"{self.source_pdf}_p{self.page_number}_c{self.chunk_index}",
            "text": self.text,
            "source_pdf": self.source_pdf,
            "page_number": self.page_number,
            "chunk_index": self.chunk_index,
        }


def estimate_tokens(text: str) -> int:
    return int(len(text.split()) * AVG_TOKENS_PER_WORD)

def split_text_block(text: str) -> List[str]:
    """Splits a long free-text block (paragraphs/notes) into token-sized pieces,
    preferring line boundaries so we don't cut mid-sentence where avoidable."""
    lines = [l for l in text.split("\n") if l.strip()]
    pieces, current = [], []
    for line in lines:
        current.append(line)
        if estimate_tokens("\n".join(current)) > MAX_TOKENS_PER_CHUNK:
            current.pop()
            if current:
                pieces.append("\n".join(current))
            current = [line]
    if current:
        pieces.append("\n".join(current))
    return pieces or [text]


SECTION_MARKER_RE = re.compile(r"^-\d+$")

def extract_title_snippet(text: str) -> str:
    """Prefers the last explicit section marker line (e.g. '-4') in the text,
    since that's what actually introduces the table that follows — falls back
    to a plain word-limited snippet if no marker is found."""
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    marker_idx = None
    for i, line in enumerate(lines):
        if SECTION_MARKER_RE.match(line):
            marker_idx = i
    snippet = " ".join(lines[marker_idx:]) if marker_idx is not None else text
    words = snippet.split()
    return " ".join(words[:TEXT_TITLE_MAX_WORDS]) if len(words) > TEXT_TITLE_MAX_WORDS else snippet

def split_markdown_pages(markdown_text: str) -> List[str]:
    return markdown_text.split("---PAGE BREAK---")


def parse_table_row(line: str) -> List[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def is_separator_line(line: str) -> bool:
    stripped = line.strip()
    return bool(re.match(r"^\|?[\s:-]+\|", stripped)) and set(stripped) <= set("|:- ")


def is_table_line(line: str) -> bool:
    return line.strip().startswith("|")


def extract_blocks(page_md: str):
    """Splits a page into ordered text blocks and table blocks (handles MULTIPLE tables per page)."""
    lines = page_md.strip().split("\n")
    blocks, pending_text, i = [], [], 0

    def flush_text():
        text = "\n".join(l for l in pending_text if l.strip())
        pending_text.clear()
        if text.strip():
            blocks.append({"type": "text", "content": text.strip()})

    while i < len(lines):
        if is_table_line(lines[i]):
            flush_text()
            table_lines = []
            while i < len(lines) and is_table_line(lines[i]):
                table_lines.append(lines[i])
                i += 1
            if len(table_lines) >= 2 and is_separator_line(table_lines[1]):
                header = parse_table_row(table_lines[0])
                rows = [parse_table_row(l) for l in table_lines[2:]]
            else:
                header, rows = None, [parse_table_row(l) for l in table_lines]
            blocks.append({"type": "table", "header": header, "rows": rows})
        else:
            pending_text.append(lines[i])
            i += 1
    flush_text()
    return blocks


def is_boilerplate_table(header, rows) -> bool:
    """Detects the repeated per-page chrome table (رقم النسخة / تاريخ الإصدار / etc.)."""
    all_cells = [c for c in (header or []) + [c for r in rows for c in r]
                 if c and not re.match(r"^~~.*~~$", c)]
    if not all_cells:
        return False
    matches = sum(1 for c in all_cells if c in BOILERPLATE_TOKENS or re.match(r"^\d{2}/\d{4}$", c))
    return matches / len(all_cells) >= 0.5


def is_role_row(row: List[str]) -> Optional[str]:
    """Detects a role-label row like ['المنفذ', 'مدير ...'] — treated as CONTEXT, not content."""
    if row and row[0].strip().rstrip(":") == "المنفذ" and len(row) > 1 and row[1].strip():
        return row[1].strip()
    return None


def render_row(row: List[str]) -> str:
    return "| " + " | ".join(row) + " |"


def render_header(header: List[str]) -> str:
    return render_row(header) + "\n| " + " | ".join("---" for _ in header) + " |"


def chunk_table_block(block, source_pdf, page_number, chunk_index_start, preceding_title=""):
    header, rows = block["header"], block["rows"]
    header_text = render_header(header) if header else ""
    chunks, chunk_index, current_role, current_rows = [], chunk_index_start, None, []

    def flush(role):
        nonlocal current_rows, chunk_index
        if not current_rows:
            return
        parts = [p for p in [preceding_title, header_text, f"({role})" if role else ""] if p]
        parts.append("\n".join(render_row(r) for r in current_rows))
        chunks.append(Chunk("\n".join(parts), source_pdf, page_number, chunk_index))
        chunk_index += 1
        current_rows = []

    for row in rows:
        role = is_role_row(row)
        if role:
            current_role = role  # role rows never become chunk content on their own
            continue
        current_rows.append(row)
        candidate = "\n".join([preceding_title, header_text, f"({current_role})" if current_role else "",
                                "\n".join(render_row(r) for r in current_rows)])
        if estimate_tokens(candidate) > MAX_TOKENS_PER_CHUNK:
            current_rows.pop()
            flush(current_role)
            current_rows.append(row)

    flush(current_role)
    return chunks, chunk_index


def chunk_document(markdown_path: str) -> List[Chunk]:
    with open(markdown_path, "r", encoding="utf-8") as f:
        full_text = f.read()

    source_pdf = os.path.splitext(os.path.basename(markdown_path))[0]
    all_chunks = []

    for page_number, page_md in enumerate(split_markdown_pages(full_text), start=1):
        chunk_index, preceding_title = 0, ""
        for block in extract_blocks(page_md):
            if block["type"] == "text":
                content = block["content"]
                word_count = len(content.split())
                if word_count > TEXT_TITLE_MAX_WORDS:
                    for piece in split_text_block(content):
                        all_chunks.append(Chunk(piece, source_pdf, page_number, chunk_index))
                        chunk_index += 1
                preceding_title = extract_title_snippet(content)
                continue
            header, rows = block["header"], block["rows"]
            if not rows and not header:
                continue
            if is_boilerplate_table(header, rows):
                continue
            table_chunks, chunk_index = chunk_table_block(
                block, source_pdf, page_number, chunk_index, preceding_title
            )
            all_chunks.extend(table_chunks)
            preceding_title = ""
    return all_chunks