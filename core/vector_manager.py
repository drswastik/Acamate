import os
import torch
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

class VectorManager:
    def __init__(self, db_dir="data/faiss_indexes"):
        self.db_dir = db_dir
        os.makedirs(self.db_dir, exist_ok=True)
        
        # 🚀 Dynamically detect CUDA for your RTX 2050
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🚀 Loading Embedding Model on {device.upper()}...")
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            model_kwargs={'device': device} 
        )

    def get_index_path(self, file_hash: str) -> str:
        """Return the directory path for a specific paper's FAISS index."""
        return os.path.join(self.db_dir, file_hash)

    def process_and_store(self, text: str, file_hash: str):
        """Chunk text, embed it, and store in FAISS. Returns the vector store."""
        index_path = self.get_index_path(file_hash)
        
        # Cache check: Don't re-embed if already done
        if os.path.exists(index_path):
            print(f"⚡ FAISS Index already exists for paper hash: {file_hash[:8]}...")
            return FAISS.load_local(
                index_path, 
                self.embeddings, 
                allow_dangerous_deserialization=True
            )

        print("✂️ Chunking academic text...")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        chunks = splitter.split_text(text)
        
        print(f"🧠 Generating embeddings for {len(chunks)} chunks...")
        vector_store = FAISS.from_texts(chunks, self.embeddings)
        
        vector_store.save_local(index_path)
        print("✅ FAISS Vector store created and saved successfully.")
        
        return vector_store

    def retrieve_top_k(self, query: str, file_hash: str, k: int = 10) -> list:
        """Retrieve the top k most relevant text chunks for a given query."""
        index_path = self.get_index_path(file_hash)
        if not os.path.exists(index_path):
            print("❌ Vector store not found. Process the PDF first.")
            return []
            
        vector_store = FAISS.load_local(
            index_path, 
            self.embeddings, 
            allow_dangerous_deserialization=True
        )
        
        docs = vector_store.similarity_search(query, k=k)
        return [doc.page_content for doc in docs]