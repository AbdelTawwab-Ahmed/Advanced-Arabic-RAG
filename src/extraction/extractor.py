import os
from dotenv import load_dotenv
from llama_parse import LlamaParse

load_dotenv()

def get_parser() -> LlamaParse:
    """Returns a configured LlamaParse instance for Arabic table extraction."""
    return LlamaParse(
        api_key=os.getenv("LLAMA_CLOUD_API_KEY"),
        result_type="markdown",
        language="ar",
        verbose=True,
    )

def extract_pdf(pdf_path: str) -> str:
    """Extracts a single PDF into a Markdown string, preserving table structure.
       Returns the combined text of all pages."""
    parser = get_parser()
    documents = parser.load_data(pdf_path)
    return "\n\n---PAGE BREAK---\n\n".join(doc.text for doc in documents)

def extract_and_save(pdf_path: str, output_dir: str = "data/processed") -> str:
    """Extracts a PDF and saves the result as a .md file named after the source PDF.
       Returns the output file path."""
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.splitext(os.path.basename(pdf_path))[0]
    output_path = os.path.join(output_dir, f"{filename}.md")

    text = extract_pdf(pdf_path)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)

    return output_path