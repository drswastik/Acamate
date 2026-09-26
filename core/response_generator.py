import hashlib
from db.db_manager import DatabaseManager
from core.query_refiner import QueryRefiner
from core.vector_manager import VectorManager
from core.context_pruner import ContextPruner
from core.mia_manager import MiAManager
from core.llm_router import HybridLLMRouter
from core.faithfulness_checker import FaithfulnessChecker
from core.memory_manager import EpisodicMemoryManager

class ResponseGenerator:
    def __init__(self, db: DatabaseManager):
        self.db = db
        self.refiner = QueryRefiner()
        self.vector_manager = VectorManager()
        self.pruner = ContextPruner()
        self.mia_manager = MiAManager(db)
        self.router = HybridLLMRouter()
        self.checker = FaithfulnessChecker()
        
        # 🧠 Initialize the new Episodic Memory RAG Engine
        self.memory = EpisodicMemoryManager(db)

    def compute_query_hash(self, paper_id: str, refined_query: str, session_id: int = 0) -> str:
        combo = f"{paper_id}:{session_id}:{refined_query}".encode("utf-8")
        return hashlib.md5(combo).hexdigest()

    def generate_response(self, paper_id, file_hash: str, session_id: int, raw_user_query: str, paper_text: str = ""):
        trace = {"raw_query": raw_user_query}
        
        # 1. RAG Gate & Semantic Routing
        route_decision = self.router.classify_intent_and_route(raw_user_query)
        
        # Hard bypass RAG if in General AI mode
        if file_hash == "GENERAL_KNOWLEDGE":
            route_decision['use_rag'] = False
            route_decision['rag_intent'] = "General Chat Mode (No PDF loaded)"
            
        trace['routing'] = route_decision
        
        # 2. Episodic Memory Retrieval (Replaces the old sliding window)
        episodic_history = self.memory.retrieve_episodic_memory(session_id, raw_user_query)
        
        if route_decision['use_rag']:
            # --- DUAL-RAG PIPELINE (PDF Search + Episodic Memory) ---
            global_map = self.mia_manager.get_or_create_mindscape(paper_id=paper_id, paper_text=paper_text)

            refined_query = self.refiner.refine_query(raw_user_query, episodic_history)
            trace['refined_query'] = refined_query
            
            raw_chunks = self.vector_manager.retrieve_top_k(refined_query, file_hash, k=12)
            pruned_chunks = self.pruner.prune_and_rerank(query=refined_query, candidate_chunks=raw_chunks, top_n=5)
            trace['pruned_chunks'] = pruned_chunks
            
            dynamic_context = "\n\n...[Snippet Break]...\n\n".join(pruned_chunks)

            prompt = f"""You are an expert academic research assistant.

=== EPISODIC CONVERSATION MEMORY ===
{episodic_history}

=== RETRIEVED RESEARCH SNIPPETS (MICRO DETAILS) ===
{dynamic_context}

=== GLOBAL MINDSCAPE MAP (MACRO CONTEXT) ===
{global_map}

INSTRUCTIONS:
1. Answer the user's latest query using facts from the RETRIEVED RESEARCH SNIPPETS.
2. Use the EPISODIC CONVERSATION MEMORY to understand references to earlier dialogue.
3. If snippets lack the information, state what is missing.
4. Do not invent facts or metrics not supported by the document.

LATEST USER QUERY: {raw_user_query}
"""
            response_text, provider = self.router.generate_routed_response(prompt, route_decision['engine'])
            score, badge = self.checker.check_faithfulness(answer=response_text, context_snippets=pruned_chunks)

            # Auto-Escalation & Safety Fallback
            if score < 0.45 and "Local LLM" in provider:
                print("\n⚠️ SYSTEM INTERCEPT: Local LLM failed faithfulness check. Auto-Escalating...")
                try:
                    response_text = self.router.gemini_model.generate_content(prompt).text.strip()
                    provider = "Gemini API (Auto-Escalated)"
                    score, badge = self.checker.check_faithfulness(answer=response_text, context_snippets=pruned_chunks)
                except Exception as e:
                    print(f"❌ Gemini Escalation failed ({e}). Triggering Emergency Local Recovery...")
                    recovery_prompt = f"""You are in emergency data-extraction mode. 
Read the following text snippets carefully:\n{dynamic_context}\n
User Question: {raw_user_query}\n
INSTRUCTIONS:
If the exact answer is present, quote it verbatim. 
If not, reply exactly: "⚠️ The local model could not find a highly factual answer, and the Gemini cloud fallback is currently unreachable."
"""
                    response_text = self.router.query_local_llm(recovery_prompt)
                    provider = "Local LLM (Emergency Recovery)"
                    score, badge = self.checker.check_faithfulness(answer=response_text, context_snippets=pruned_chunks)
                    
        else:
            # --- SINGLE-RAG PIPELINE (Episodic Memory ONLY - No PDF pulled) ---
            trace['refined_query'] = "N/A (Bypassed PDF RAG via Gate)"
            trace['pruned_chunks'] = []
            
            prompt = f"""You are a helpful and intelligent AI assistant.

=== EPISODIC CONVERSATION MEMORY ===
{episodic_history}

INSTRUCTIONS:
You are maintaining continuous context for this ongoing conversation. Answer the latest query naturally and thoughtfully.

LATEST USER QUERY: {raw_user_query}
"""
            response_text, provider = self.router.generate_routed_response(prompt, route_decision['engine'])
            score, badge = 1.0, "☑️ General Chat (Episodic Memory Active)"
            
        trace['final_provider'] = provider
        trace['faithfulness_badge'] = badge
        return response_text, provider, score, badge, trace

    def get_response(self, paper_id, file_hash: str, session_id: int, user_query: str, paper_text: str = "") -> dict:
        print("\n" + "=" * 50)
        print(f"🚀 Processing Query: '{user_query}'")
        print("=" * 50)

        query_hash = self.compute_query_hash(str(paper_id), user_query, session_id)

        cached = self.db.get_cached_response(query_hash)
        if cached:
            print("⚡ Cache Hit — Returning stored verified result!")
            answer = cached["response"]
            provider = f"{cached['routing_provider']} (Cached)"
            score = cached["faithfulness_score"]
            badge = "⚡ Verified Cache Hit"
            trace = {"routing": {"rag_intent": "Fetched from Cache", "use_rag": False, "engine": "Cache"}}
            
            self.db.add_chat_message(session_id=session_id, role="user", content=user_query)
            self.db.add_chat_message(session_id=session_id, role="assistant", content=answer, routing_provider=provider, faithfulness_score=score)
        else:
            answer, provider, score, badge, trace = self.generate_response(
                paper_id, file_hash, session_id, user_query, paper_text
            )
            
            self.db.add_chat_message(session_id=session_id, role="user", content=user_query)
            self.db.add_chat_message(
                session_id=session_id, role="assistant", content=answer, 
                routing_provider=provider, faithfulness_score=score
            )
            
            if paper_id and file_hash != "GENERAL_KNOWLEDGE": 
                self.db.insert_query_response(
                    paper_id=paper_id, query_hash=query_hash, user_query=user_query,
                    refined_query=trace.get('refined_query', ''), response=answer, provider=provider, faithfulness_score=score
                )

        return {
            "answer": answer, "provider": provider, 
            "faithfulness_score": score, "badge": badge, 
            "query_hash": query_hash, "trace": trace
        }