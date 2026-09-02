import os
from src.extraction.extractor import extract_and_save

RAW_DIR = "data/raw"

if __name__ == "__main__":
    pdf_files = [f for f in os.listdir(RAW_DIR) if f.endswith(".pdf")]
    for pdf_file in pdf_files:
        path = os.path.join(RAW_DIR, pdf_file)
        output_path = extract_and_save(path)
        print(f"\nExtracted: {pdf_file} -> {output_path}")