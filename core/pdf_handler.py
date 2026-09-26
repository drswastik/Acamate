import os
import re
import hashlib
import pdfplumber
from db.db_manager import DatabaseManager
from core.vector_manager import VectorManager  # 👈 NEW IMPORT

class PDFHandler:
    def __init__(self, db: DatabaseManager):
        self.db = db
        self.vector_manager = VectorManager()  # 👈 INSTANTIATE HERE

    # ... keep your existing extract_text, clean_text, get_file_hash methods ...
    def extract_text(self, file_path):
        """Extract text from PDF using pdfplumber."""
        text = ""
        try:
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    text += page.extract_text() or ""
            return text.strip()
        except Exception as e:
            print(f"❌ Error reading PDF: {e}")
            return ""

    def clean_text(self, text):
        """Clean extracted text using regex."""
        text = re.sub(r'\n+', '\n', text)
        text = re.sub(r'\s{2,}', ' ', text)
        text = re.sub(r'[^\x00-\x7F]+', '', text)
        return text.strip()

    def get_file_hash(self, file_path):
        """Generate MD5 hash of file contents."""
        hasher = hashlib.md5()
        try:
            with open(file_path, 'rb') as f:
                buf = f.read()
                hasher.update(buf)
            return hasher.hexdigest()
        except Exception as e:
            print(f"❌ Error generating hash: {e}")
            return None

    def process_pdf(self, file_path):
        """Full pipeline: extract, clean, hash, store, and EMBED."""
        if not os.path.exists(file_path):
            print("❌ File not found!")
            return None

        file_name = os.path.basename(file_path)
        file_hash = self.get_file_hash(file_path)
        
        # We need the cleaned text for FAISS regardless of whether the DB knows the paper
        text = self.extract_text(file_path)
        clean = self.clean_text(text)

        # 🚀 THE UPGRADE: Build FAISS embeddings silently using the GPU
        print("\n⚙️  Processing document geometry and vectors...")
        self.vector_manager.process_and_store(clean, file_hash)

        # Check if already in DB
        paper_id = self.db.get_paper_id(file_hash)
        if paper_id:
            print(f"✅ Paper already exists in DB (ID: {paper_id})")
            return {
                "paper_id": paper_id,
                "file_name": file_name,
                "file_hash": file_hash,
                "text": clean
            }

        # Store paper metadata
        self.db.insert_paper(file_name, file_hash, file_path)
        paper_id = self.db.get_paper_id(file_hash)

        print(f"✅ PDF processed successfully! Stored as Paper ID: {paper_id}")
        return {
            "paper_id": paper_id,
            "file_name": file_name,
            "file_hash": file_hash,
            "text": clean
        }