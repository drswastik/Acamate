import faiss
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from db.db_manager import DatabaseManager

class EpisodicMemoryManager:
    def __init__(self, db: DatabaseManager):
        self.db = db
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🧠 Loading Episodic Memory RAG (all-MiniLM-L6-v2) on {self.device.upper()}...")
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2", device=self.device)
        self.dimension = 384

    def retrieve_episodic_memory(self, session_id: int, query: str, k: int = 3) -> str:
        """
        Retrieves relevant past conversation turns using FAISS Vector Search.
        Always retains the most recent 2 turns for immediate conversational flow.
        """
        # Fetch a deep history (up to 200 turns)
        raw_history = self.db.get_chat_history(session_id, limit=200) 
        
        # If history is very short, no need for RAG, just return it directly
        if len(raw_history) <= 4:
            formatted_turns = []
            for msg in raw_history:
                role = "User" if msg["role"] == "user" else "Assistant"
                formatted_turns.append(f"{role}: {msg['content']}")
            return "\n".join(formatted_turns)
            
        # 1. Format all historical turns
        episodes = []
        for msg in raw_history:
            role = "User" if msg["role"] == "user" else "Assistant"
            episodes.append(f"{role}: {msg['content']}")
            
        # 2. Build on-the-fly FAISS index for this session's memory
        index = faiss.IndexFlatIP(self.dimension)
        embeddings = self.encoder.encode(episodes, normalize_embeddings=True, convert_to_numpy=True)
        index.add(embeddings)
        
        # 3. Query the Memory Index with the user's latest message
        query_emb = self.encoder.encode([query], normalize_embeddings=True, convert_to_numpy=True)
        distances, indices = index.search(query_emb, k)
        
        # 4. Extract recent and recalled turns
        recent_turns = episodes[-2:] # Always keep the last 2 turns intact
        
        retrieved_turns = []
        # Sort indices chronologically to maintain natural reading order
        sorted_indices = sorted([idx for idx in indices[0] if idx != -1])
        
        for idx in sorted_indices:
            # Prevent duplicating the recent turns if they were matched by FAISS
            if episodes[idx] not in recent_turns:
                retrieved_turns.append(f"[Turn {idx+1}] {episodes[idx]}")
                
        # 5. Merge Contexts Seamlessly
        context_parts = []
        if retrieved_turns:
            context_parts.append("[Recalled Past Memory]:\n" + "\n".join(retrieved_turns))
        context_parts.append("[Immediate Recent Context]:\n" + "\n".join(recent_turns))
        
        return "\n\n".join(context_parts)