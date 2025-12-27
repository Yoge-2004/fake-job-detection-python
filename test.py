import joblib
import spacy
import re
import numpy as np
import os
from sklearn.base import BaseEstimator, TransformerMixin
from lime.lime_text import LimeTextExplainer

# --- CONFIGURATION ---
MODEL_FILE_PATH = 'production_fake_job_pipeline.pkl'
SPACY_MODEL_NAME = 'en_core_web_lg'


# ---------------------

# ==========================================
# 1. CUSTOM CLASSES
# ==========================================
class TextCleaner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self

    def transform(self, X):
        cleaned = []
        for text in X:
            text = str(text).lower() if text else ""
            text = re.sub(r'http\S+|www\.\S+', 'token_url', text)
            text = re.sub(r'\S+@\S+', 'token_email', text)
            text = re.sub(r'[^a-z0-9\s\$\%\@\.\,\!]', '', text)
            text = re.sub(r'\s+', ' ', text).strip()
            cleaned.append(text if text else "token_empty_input")
        return cleaned


class SpacyVectorTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, nlp=None):
        self.nlp = nlp

    def fit(self, X, y=None): return self

    def transform(self, X):
        if self.nlp is None:
            import spacy
            self.nlp = spacy.load("en_core_web_md")
        docs = list(self.nlp.pipe(X, disable=["ner", "parser"]))
        return np.array([doc.vector if doc.has_vector else np.zeros(300) for doc in docs])


# ==========================================
# 2. THE HUMAN REASONING ENGINE 🧠
# ==========================================
def explain_like_a_human(text, label, lime_features):
    """
    Translates technical signals into human-readable bullet points.
    """
    reasons = []
    text_lower = text.lower()

    # --- 1. LOGIC GAPS (The Detective Layer) ---
    if "usa" in text_lower or "united states" in text_lower or "california" in text_lower:
        if "lpa" in text_lower or "rupees" in text_lower:
            reasons.append("⚠️ **Geography Mismatch:** Location is USA/Global, but salary is in Indian Currency (LPA).")

    if "softwares" in text_lower:
        reasons.append(
            "⚠️ **Grammar Red Flag:** Uses 'softwares' (incorrect plural), often a sign of unprofessional scams.")

    if "kindly" in text_lower and "pay" in text_lower:
        reasons.append("⚠️ **Scam Phrasing:** 'Kindly pay' is a common phrase used in payment fraud.")

    # --- 2. DANGEROUS KEYWORDS (The Trigger Layer) ---
    triggers = {
        "telegram": "🚨 **Off-Platform Risk:** Asks to move chat to Telegram (High Fraud Risk).",
        "whatsapp": "🚨 **Off-Platform Risk:** Asks to move chat to WhatsApp.",
        "usdt": "💸 **Crypto Risk:** Mentions USDT/Crypto payments (Likely Money Laundering).",
        "check": "💸 **Payment Risk:** Mentions sending a 'Check' (Likely Mobile Deposit Fraud).",
        "training fee": "💸 **Upfront Cost:** Asks for money/fees before hiring (Illegal in most places).",
        "gmail.com": "⚠️ **Generic Email:** Uses a public domain (@gmail) instead of a company email."
    }

    for word, reason in triggers.items():
        if word in text_lower:
            reasons.append(reason)

    # --- 3. LIME EXPLAINER (The AI Layer) ---
    # If no specific rules triggered, look at what the AI found suspicious
    if not reasons and label == "FAKE":
        top_words = [w[0] for w in lime_features if w[1] > 0][:3]
        if top_words:
            reasons.append(
                f"🤖 **AI Pattern Match:** The model found unusual clustering around words: {', '.join(top_words)}.")

    if label == "REAL" and not reasons:
        reasons.append("✅ **Corporate Standard:** Contains professional terminology (e.g., specific skills, benefits).")
        reasons.append("✅ **No Triggers:** No high-risk fraud keywords detected.")

    return reasons


# ==========================================
# 3. CORE PIPELINE LOGIC
# ==========================================
def load_resources():
    print("--- Initializing AI Analyst ---")
    try:
        nlp_engine = spacy.load(SPACY_MODEL_NAME)
    except:
        from spacy.cli import download
        download(SPACY_MODEL_NAME)
        nlp_engine = spacy.load(SPACY_MODEL_NAME)

    if not os.path.exists(MODEL_FILE_PATH):
        print("❌ Model not found.")
        return None, None

    with open(MODEL_FILE_PATH, 'rb') as f:
        pipeline = joblib.load(f)

    # Inject NLP
    def inject(est):
        if isinstance(est, SpacyVectorTransformer): est.nlp = nlp_engine; return True
        if hasattr(est, 'steps'): [inject(s[1]) for s in est.steps]
        if hasattr(est, 'transformer_list'): [inject(s[1]) for s in est.transformer_list]
        if hasattr(est, 'estimator'): inject(est.estimator)

    inject(pipeline)
    print("✅ System Ready.")
    return nlp_engine, pipeline


def analyze_job_post(text, pipeline, nlp_engine):
    # --- GUARDS (Joker/Matrix) ---
    doc = nlp_engine(text)
    valid_tokens = [t for t in doc if t.is_alpha and t.has_vector]
    total_tokens = [t for t in doc if t.is_alpha]
    valid_ratio = len(valid_tokens) / len(total_tokens) if total_tokens else 0

    if valid_ratio < 0.4:
        return 0.99, "FAKE", ["🚫 **Gibberish Detected:** Text contains mostly non-English or random words."], []

    if any(t in text.upper() for t in ["SELECT *", "DROP TABLE", "<SCRIPT>"]):
        return 0.99, "FAKE", ["🚨 **Security Threat:** Malicious Code / SQL Injection detected."], []

    # --- PREDICTION ---
    max_prob = pipeline.predict_proba([text])[0][1]

    # --- XAI ---
    xai_data = []
    try:
        if len(text.split()) > 5:
            explainer = LimeTextExplainer(class_names=['REAL', 'FAKE'])
            exp = explainer.explain_instance(text, pipeline.predict_proba, num_features=5, num_samples=100)
            xai_data = exp.as_list()
    except:
        pass

    # --- FINAL VERDICT ---
    label = "FAKE" if max_prob > 0.5 else "REAL"

    # Get Human Explanations
    human_reasons = explain_like_a_human(text, label, xai_data)

    return max_prob, label, human_reasons, xai_data


# ==========================================
# 4. INTERACTIVE CONSOLE
# ==========================================
if __name__ == "__main__":
    nlp_engine, loaded_pipeline = load_resources()

    if loaded_pipeline:
        print("\n" + "=" * 60)
        print("🕵️  AI JOB ANALYST (Human-Readable Explanations)")
        print("Type 'QUIT' to exit.")
        print("=" * 60)

        while True:
            try:
                print("\n" + "-" * 60)
                user_input = input("Paste Job Description: ")
                if user_input.strip().upper() in ['QUIT', 'EXIT']: break
                if not user_input.strip(): continue

                print("⏳ Analyzing...")
                prob, label, reasons, xai_data = analyze_job_post(user_input, loaded_pipeline, nlp_engine)

                print("\n" + "=" * 40)
                print(f"   VERDICT: {label}")
                print("=" * 40)

                # SCORE
                if label == "FAKE":
                    print(f"Risk Level:    {prob * 100:.1f}% 🔴 (High Risk)")
                else:
                    print(f"Safety Score:  {(1 - prob) * 100:.1f}% 🟢 (Safe)")

                # HUMAN EXPLANATION
                print("\n📝 WHY DID WE SAY THIS?")
                print("-" * 40)
                for reason in reasons:
                    print(f"  {reason}")

                # XAI CHART
                if xai_data:
                    print("\n🧠 AI WORD WEIGHTS (LIME)")
                    print("-" * 40)
                    for word, score in xai_data:
                        impact = "🔴 FAKE" if score > 0 else "🟢 REAL"
                        print(f"  {word:<15} | {impact} ({score:+.2f})")

            except Exception as e:
                print(f"Error: {e}")
