import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class QueryRefiner:
    def __init__(self, model_name="D:\\Acamate\\local_flan_t5_small"):
        # If your model is in a local folder, you can change model_name to your folder path!
        # e.g., model_name = "./flan-t5-small"
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"✨ Loading Query Refiner ({model_name}) on {self.device.upper()}...")
        
        # 🐛 FIX: Bypassing the volatile Pipeline API and loading the model natively 
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)

    def refine_query(self, user_query: str, chat_history: str = "") -> str:
        """Uses Flan-T5 to rewrite vague conversational queries into specific search keywords."""
        if len(user_query.split()) > 10:
            return user_query  # Already specific enough

        print("✨ Refining query locally via Flan-T5-small...")
        
        prompt = f"Rewrite this query into specific search keywords based on context. Context: {chat_history[-200:]} Query: {user_query}"

        try:
            # Tokenize the prompt and move to GPU/CPU
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to(self.device)
            
            # Generate the output natively using max_new_tokens (the proper way for T5)
            outputs = self.model.generate(**inputs, max_new_tokens=50)
            
            # Decode the generated tokens back into a string
            refined = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            
            if not refined:
                return user_query
                
            print(f"🎯 Refined: '{refined}'")
            return refined
        except Exception as e:
            print(f"⚠️ Flan-T5 refinement failed ({e}). Using raw query.")
            return user_query

    def summarize_history(self, history_text: str) -> str:
        """Uses Flan-T5 to compress older chat history for Lossless Memory."""
        prompt = f"Briefly summarize this conversation: {history_text}"
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(self.device)
            outputs = self.model.generate(**inputs, max_new_tokens=80)
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        except:
            return "Previous context retained."