# store_index.py
from dotenv import load_dotenv
import os
from src.helper import load_pdf_file, filter_to_minimal_docs, text_split, download_hugging_face_embeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

load_dotenv()

# -----------------------
# Environment variables
# -----------------------
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    raise SystemExit("PINECONE_API_KEY missing in .env — required to store embeddings.")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# -----------------------
# Load and process PDFs
# -----------------------
print("Loading PDFs from './data'...")
extracted_data = load_pdf_file(path="./data")
print(f"Loaded {len(extracted_data)} PDFs.")

filtered_data = filter_to_minimal_docs(extracted_data)
text_chunks = text_split(filtered_data)
print(f"Created {len(text_chunks)} text chunks.")

# -----------------------
# Load embeddings
# -----------------------
print("Loading HuggingFace embeddings...")
embeddings = download_hugging_face_embeddings()
print("Embeddings loaded.")

# -----------------------
# Pinecone setup
# -----------------------
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "puc-chatbot"

if not pc.has_index(index_name):
    print(f"Creating Pinecone index '{index_name}'...")
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    print(f"Index '{index_name}' created successfully.")
else:
    print(f"Using existing Pinecone index '{index_name}'.")

index = pc.Index(index_name)

# -----------------------
# Insert documents into Pinecone
# -----------------------
print("Uploading documents to Pinecone...")
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding=embeddings,
)

# -----------------------
# Verify insertion
# -----------------------
stats = index.describe_index_stats()
print("Index stats after insertion:", stats)
print("✅ Pinecone upload complete.")
