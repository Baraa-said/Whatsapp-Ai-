
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.document_loader import DocumentLoader
from src.vector_store import VectorStore

try:
    print("Testing DocumentLoader...")
    default_file = "phone_store_inventory.md"
    if not os.path.exists(default_file):
        print(f"Error: {default_file} not found!")
        sys.exit(1)

    loader = DocumentLoader()
    print("Loader initialized.")
    
    documents = loader.load_and_split([default_file])
    print(f"Documents loaded: {len(documents)}")
    for doc in documents:
        print(f"Content preview: {doc.page_content[:50]}...")

    print("Testing VectorStore creation...")
    vs = VectorStore()
    vs.create_from_documents(documents)
    print("VectorStore created successfully.")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
