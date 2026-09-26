import torch
import re
from sentence_transformers import CrossEncoder

class FaithfulnessChecker:
    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-small", threshold: float = 0.5):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🛡️ Loading NLI Hallucination Guardrail on {self.device.upper()}...")
        
        self.model = CrossEncoder(model_name, device=self.device)
        self.threshold = threshold

    def check_faithfulness(self, answer: str, context_snippets: list[str]) -> tuple[float, str]:
        if not answer or not context_snippets:
            return 0.0, "[⚠️ Unverified: Missing Context]"

        if "NOT_IN_CONTEXT" in answer or "does not contain" in answer.lower():
            return 1.0, "ℹ️ Info Not in Text (Verified Refusal)"

        # Limit context to avoid breaking the 512-token DeBERTa limit
        combined_context = " ".join(context_snippets)[:2000]

        # 🧠 INTELLIGENT ARCHITECTURE: Sentence-Level Evaluation
        sentences = re.split(r'(?<=[.!?]) +', answer)
        
        # Filter out extremely short fragments or conversational filler
        sentences = [s.strip() for s in sentences if len(s.strip()) > 15]

        if not sentences:
            return 0.0, "⚠️ Could not parse answer for evaluation."

        sentence_scores = []
        
        for sentence in sentences:
            pair = [combined_context, sentence]
            logits = self.model.predict([pair])[0]
            
            exp_logits = torch.exp(torch.tensor(logits))
            probs = exp_logits / torch.sum(exp_logits)
            
            # Entailment is index 1
            sentence_scores.append(float(probs[1]))

        # Calculate the true average entailment score of the actual facts
        avg_score = sum(sentence_scores) / len(sentence_scores)

        if avg_score >= 0.70:
            badge = f"✅ Highly Verified (Score: {avg_score:.2f})"
        elif avg_score >= self.threshold:
            badge = f"☑️ Partially Verified (Score: {avg_score:.2f})"
        else:
            badge = f"⚠️ Low Faithfulness (Score: {avg_score:.2f})"

        print(f"🛡️ Sentence-Level Audit Complete ──► {badge}")
        return avg_score, badge