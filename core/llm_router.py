import torch
import requests
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from config.config import GEMINI_API_KEY

class HybridLLMRouter:
    def __init__(self, ollama_model: str = "llama3.1", ollama_url: str = "http://localhost:11434/api/generate"):
        self.ollama_model = ollama_model
        self.ollama_url = ollama_url
        
        # Configure Gemini
        genai.configure(api_key=GEMINI_API_KEY)
        self.gemini_model = genai.GenerativeModel("models/gemini-flash-lite-latest")

        # Dense Semantic Vector Router
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🔀 Loading Dense Semantic Router (all-MiniLM-L6-v2) on {self.device.upper()}...")
        
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2", device=self.device)
        
        # Define semantic anchors (prototypical sentences for each intent)
        self.rag_anchors = {
            "USE_RAG": [
                "search the document", "what does the paper say", 
                "explain the methodology in the pdf", "summarize the text",
                "according to the authors", "use the document context"
            ],
            "NO_RAG": [
                "hello", "use online model", "use local model", "write a python script", 
                "what is quantum computing", "route via online model", "switch to ollama",
                "use some other on device model please"
            ]
        }
        
        self.engine_anchors = {
            "CLOUD": [
                "use online model", "route via online model", "use gemini", 
                "use the cloud model", "use the best model available", 
                "switch to online", "use the api"
            ],
            "LOCAL": [
                "use local model", "use some other on device model please", 
                "use local llama", "use offline model", "run locally", 
                "switch to ollama", "keep it local", "use some other model please"
            ]
        }
        
        # Pre-compute tensors in memory
        self.rag_tensors = {k: self.encoder.encode(v, convert_to_tensor=True) for k, v in self.rag_anchors.items()}
        self.engine_tensors = {k: self.encoder.encode(v, convert_to_tensor=True) for k, v in self.engine_anchors.items()}

    def _get_highest_similarity(self, query_emb, anchor_tensors) -> float:
        cos_scores = torch.nn.functional.cosine_similarity(query_emb.unsqueeze(0), anchor_tensors)
        return cos_scores.max().item()

    def classify_intent_and_route(self, query: str) -> dict:
        query_emb = self.encoder.encode(query, convert_to_tensor=True)

        score_use_rag = self._get_highest_similarity(query_emb, self.rag_tensors["USE_RAG"])
        score_no_rag = self._get_highest_similarity(query_emb, self.rag_tensors["NO_RAG"])
        
        score_cloud = self._get_highest_similarity(query_emb, self.engine_tensors["CLOUD"])
        score_local = self._get_highest_similarity(query_emb, self.engine_tensors["LOCAL"])

        print(f"🔍 [Semantic Math] RAG({score_use_rag:.2f}) vs NO_RAG({score_no_rag:.2f})")
        print(f"🔍 [Semantic Math] CLOUD({score_cloud:.2f}) vs LOCAL({score_local:.2f})")

        use_rag = score_use_rag > score_no_rag
        
        if score_cloud > 0.50 and score_cloud > score_local:
            engine = "GEMINI"
        else:
            engine = "LOCAL"
            
        rag_intent = "PDF Document Search" if use_rag else "General Chat / Meta Command"

        print(f"🧠 Vector Router Decision: RAG={use_rag}, Engine={engine}")
        return {
            "use_rag": use_rag,
            "rag_intent": rag_intent,
            "engine": engine
        }

    def query_local_llm(self, prompt: str) -> str:
        payload = {
            "model": self.ollama_model,
            "prompt": prompt,
            "stream": False
        }
        try:
            response = requests.post(self.ollama_url, json=payload, timeout=120)
            if response.status_code == 200:
                return response.json().get("response", "").strip()
            return f"⚠️ Ollama error HTTP {response.status_code}."
        except Exception as e:
            return f"⚠️ Could not reach local Ollama server."

    def generate_routed_response(self, prompt: str, engine: str = None, **kwargs) -> tuple[str, str]:
        # Backward compatibility for core/mia_manager.py
        # It passes `user_query` instead of `engine`. We catch it with **kwargs
        if engine is None:
            if "user_query" in kwargs:
                engine = "GEMINI"  # Large PDF Mindscape summaries are handled by Gemini
            else:
                engine = "LOCAL"
                
        # Catch-all fallback
        if engine not in ["LOCAL", "GEMINI"]:
            engine = "GEMINI"

        if engine == "LOCAL":
            answer = self.query_local_llm(prompt)
            return answer, "Local LLM (Ollama)"
        else:
            try:
                response = self.gemini_model.generate_content(prompt)
                return response.text.strip(), "Gemini API (Flash-Lite)"
            except Exception as e:
                error_msg = str(e).lower()
                if "quota" in error_msg or "429" in error_msg:
                    print("❌ Gemini API Rate Limit Exceeded.")
                    fallback = self.query_local_llm(prompt)
                    return f"**[Gemini API Quota Exceeded. Local Fallback Used.]**\n\n{fallback}", "Local LLM (Gemini Fallback)"
                else:
                    return f"❌ Gemini API Error: {e}", "Error"