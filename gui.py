import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
from google import genai

st.set_page_config(page_title="PDF Vector Search System", layout="wide")
st.title("Context-Aware Document Intelligence System")
st.subheader("RAG Pipeline using Chroma DB & Gemini API")

# configure Gemini API Key
api_key = st.sidebar.text_input("Enter Gemini API Key:", type="password")

@st.cache_resource
def get_collection():
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    return chroma_client.get_collection(
        name="pdf_collection", 
        embedding_function=embedding_fn
    )

try:
    collection = get_collection()
    st.success("Connected to local Chroma DB index!")
except Exception as e:
    st.error("Vector database index not found.")
    st.stop()

query = st.text_input("Enter your question about the document:")
top_k = st.slider("Number of chunks to retrieve (top-k):", 1, 10, 3)

if st.button("Search & Generate Answer") and query:
    with st.spinner("Retrieving relevant contexts from Chroma DB..."):
        results = collection.query(
            query_texts=[query],
            n_results=top_k
        )
    
    # combine contexts
    retrieved_texts = []
    st.write("### 1. Retrieved DB Contexts:")
    for i, doc in enumerate(results["documents"][0]):
        page = results["metadatas"][0][i].get("page", "N/A")
        retrieved_texts.append(f"[Page {page}]: {doc}")
        with st.expander(f"Result {i+1} | Page Number: {page}"):
            st.write(doc)
            
    # generate answer if API key is provided
    if api_key:
        with st.spinner("Generating synthesized answer using Gemini..."):
            client = genai.Client(api_key=api_key)
            context_block = "\n\n".join(retrieved_texts)
            prompt = f"""Use the following retrieved contexts from the document to answer the question accurately.
            
Contexts:
{context_block}

Question: {query}

Answer in simple, clear points and cite the page numbers used:"""
            
            response = client.models.generate_content(
                model="gemini-3.8-flash",
                contents=prompt
            )
            
            st.write("### 2. Generated LLM Answer:")
            st.info(response.text)
    else:
        st.warning("Enter a Gemini API Key in the sidebar to generate synthesized answers.")