import os
from db.db_manager import DatabaseManager
from core.llm_router import HybridLLMRouter

class MiAManager:
    def __init__(self, db: DatabaseManager):
        self.db = db
        self.router = HybridLLMRouter()

    def get_or_create_mindscape(self, paper_id: int, paper_text: str) -> str:
        """
        Fetches the existing Mindscape Map from SQLite or generates a new Global Summary
        representing the macro-level view of the entire paper.
        """
        # 1. Check SQLite DB cache first
        cached_map = self.db.get_mia_map(paper_id)
        if cached_map:
            print("⚡ MiA Global Mindscape Map fetched from DB cache!")
            return cached_map["global_summary"]

        print("🗺️ Generating new MiA Global Mindscape Map...")

        # 2. Extract representative sections (Intro + Middle + Conclusion)
        text_len = len(paper_text)
        if text_len <= 4000:
            sample_text = paper_text
        else:
            intro = paper_text[:2000]
            middle = paper_text[text_len // 2 : (text_len // 2) + 1500]
            conclusion = paper_text[-1500:]
            sample_text = (
                f"{intro}\n\n"
                f"[...Middle Section...]\n\n"
                f"{middle}\n\n"
                f"[...Conclusion & Results...]\n\n"
                f"{conclusion}"
            )

        prompt = f"""
You are an expert academic research strategist.
Create a concise, structured "Global Mindscape Map" for the following research paper context.

This map will serve as global background memory for future detailed queries.

Include the following structured bullet points:
• Core Research Problem & Objective
• Primary Methodology / System Architecture
• Key Experimental Results & Metrics
• Scope, Assumptions & Main Limitations

Paper Context:
{sample_text}

Structured Global Mindscape Map:
"""
        # 3. Route generation via Hybrid Router (uses Local Llama 3.1 to save API costs)
        global_summary, provider = self.router.generate_routed_response(
            prompt=prompt, 
            user_query="summarize whole paper global mindscape map"
        )

        # 4. Save to SQLite database so we never compute this paper's macro view again
        self.db.insert_mia_map(paper_id=paper_id, global_summary=global_summary)
        print(f"✅ Created and persisted MiA Mindscape Map via {provider}.")

        return global_summary


# test
if __name__ == "__main__":
    db = DatabaseManager()
    db.create_tables()

    mia = MiAManager(db)
    dummy_text = "Abstract: This paper introduces a novel power grid stabilizer... " * 100
    
    # Simulate paper ID 1
    summary = mia.get_or_create_mindscape(paper_id=1, paper_text=dummy_text)
    print("\n🗺️ Generated Mindscape Summary:\n", summary)
    
    db.close()