import torch
from sentence_transformers import CrossEncoder

class ContextPruner:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", min_score_threshold: float = -2.0):
        # Determine device (forces onto CUDA if available)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🎯 Loading Cross-Encoder Pruner on {self.device.upper()}...")
        
        self.model = CrossEncoder(model_name, device=self.device)
        self.min_score_threshold = min_score_threshold

    def prune_and_rerank(self, query: str, candidate_chunks: list[str], top_n: int = 3) -> list[str]:
        """
        Takes raw candidate chunks from FAISS, evaluates their deep semantic relevance 
        against the query, prunes noise, and returns the top_n highest scoring chunks.
        """
        if not candidate_chunks:
            return []

        # Construct pairs: [ [query, chunk_1], [query, chunk_2], ... ]
        pairs = [[query, chunk] for chunk in candidate_chunks]

        # Get Cross-Encoder relevance scores
        scores = self.model.predict(pairs)

        # Pair up chunks with their respective score
        scored_chunks = list(zip(candidate_chunks, scores))

        # Filter out low-confidence noise below threshold
        filtered_chunks = [pair for pair in scored_chunks if pair[1] >= self.min_score_threshold]

        # Fallback if threshold was too aggressive
        if not filtered_chunks:
            filtered_chunks = scored_chunks

        # Sort descending by Cross-Encoder score
        filtered_chunks.sort(key=lambda x: x[1], reverse=True)

        selected_chunks = [chunk for chunk, score in filtered_chunks[:top_n]]

        print(f"✂️  Pruned {len(candidate_chunks)} raw candidate chunks ──► {len(selected_chunks)} high-density chunks.")
        return selected_chunks

# test
if __name__ == "__main__":
    pruner = ContextPruner()
    sample_query = "What optimizer was used in training?"
    sample_chunks = [
        "We trained the model for 50 epochs on an NVIDIA A100 GPU cluster.",
        "The AdamW optimizer was utilized with a learning rate of 1e-4 and weight decay of 0.01.",
        "The dataset consists of over 100,000 academic research papers collected in 2024.",
        "Gradient clipping was set to 1.0 to prevent exploding gradients during backpropagation."
    ]

    best_chunks = pruner.prune_and_rerank(sample_query, sample_chunks, top_n=2)
    print("\n🔍 Top Pruned Result:\n", best_chunks[0])