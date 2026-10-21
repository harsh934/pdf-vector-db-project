import chromadb
from chromadb.utils import embedding_functions
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. load the pdf document
print("loading document.pdf...")
loader = PyPDFLoader("document.pdf")
docs = loader.load()
print(f"loaded {len(docs)} pages.")

# 2. chunk the text into smaller segments
print("chunking text into smaller segments...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=150
)
chunks = text_splitter.split_documents(docs)
print(f"total chunks created: {len(chunks)}")

# 3. setup local persistent chroma db
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# sentence-transformers embedding function
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

collection = chroma_client.get_or_create_collection(
    name="pdf_collection",
    embedding_function=embedding_fn
)

# 4. store chunks and vectors in chroma db
documents = [chunk.page_content for chunk in chunks]
metadatas = [chunk.metadata for chunk in chunks]
ids = [f"id_{i}" for i in range(len(chunks))]

print("generating embeddings and storing in chroma db...")
collection.add(
    documents=documents,
    metadatas=metadatas,
    ids=ids
)

print("\n--- ingestion complete! ---")

# 5. test similarity query
user_query = "what is the main message or key takeaway of this book?"
print(f"\ntesting query: '{user_query}'\n")

results = collection.query(
    query_texts=[user_query],
    n_results=3
)

for i, doc in enumerate(results["documents"][0]):
    page = results["metadatas"][0][i].get("page", "N/A")
    print(f"--- result {i+1} | page {page} ---")
    print(f"{doc}\n")