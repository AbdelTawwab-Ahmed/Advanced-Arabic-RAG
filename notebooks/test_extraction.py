import os
from dotenv import load_dotenv
from llama_parse import LlamaParse

load_dotenv()

parser = LlamaParse(
    api_key=os.getenv("LLAMA_CLOUD_API_KEY"),
    result_type="markdown",   
    language="ar",            
    verbose=True,
)

documents = parser.load_data("data/raw/Central Alarm Tasks And Procedures Manual.pdf")

os.makedirs("data/processed", exist_ok=True)
with open("data/processed/test_extraction_output.md", "w", encoding="utf-8") as f:
    for doc in documents:
        f.write(doc.text)
        f.write("\n\n---PAGE BREAK---\n\n")

print(f"\nExtracted {len(documents)} document object(s). Output saved to data/processed/test_extraction_output.md")