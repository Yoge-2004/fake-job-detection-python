import warnings
# warnings.filterwarnings("ignore")
from flask import Flask, render_template, request, jsonify, session, redirect, url_for, g
from flask_caching import Cache
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
import joblib
import os
import re
import numpy as np
import sqlite3
import sys
import hashlib
from datetime import datetime, timedelta
from werkzeug.security import generate_password_hash, check_password_hash
from lime.lime_text import LimeTextExplainer
from flashtext import KeywordProcessor

# --- TENSORFLOW FOR UNSUPERVISED MODEL ---
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization

app = Flask(__name__)
app.secret_key = "jobguard_super_secret_key"
app.permanent_session_lifetime = timedelta(days=30)
DB_NAME = "users.db"

# ==========================================
# 0. CACHE & LOGGING CONFIG
# ==========================================
cache_config = {
    "CACHE_TYPE": "SimpleCache",
    "CACHE_DEFAULT_TIMEOUT": 3600
}
app.config.from_mapping(cache_config)
cache = Cache(app)

SERVER_LOGS = []

def log_debug(message, level="INFO"):
    timestamp = datetime.now().strftime("%H:%M:%S")
    entry = f"[{timestamp}] {level}: {message}"
    SERVER_LOGS.append(entry)
    if len(SERVER_LOGS) > 100: SERVER_LOGS.pop(0)
    print(entry, flush=True)

# ==========================================
# 1. REGEX & FLASHTEXT SETUP
# ==========================================
REGEX_LONG_STRING = re.compile(r'\S{40,}')
REGEX_REPEATED_CHARS = re.compile(r'(.)\1{4,}')
REGEX_CONSONANT_SMASH = re.compile(r'[aeiouy]')
REGEX_HTTP = re.compile(r'http\S+|www\.\S+')
REGEX_EMAIL = re.compile(r'\S+@\S+')
REGEX_SPECIAL_CHARS = re.compile(r'[^a-z0-9\s\$\%\@\.\,\!]')
REGEX_SPACES = re.compile(r'\s+')
REGEX_WORD_TOKEN = re.compile(r'\w+')
REGEX_CODE_SNIPPET = re.compile(r'(;|{|}|\<script\>|SELECT \*|DROP TABLE)')

# --- FLASHTEXT (HARDCODED RED FLAGS) ---
scam_processor = KeywordProcessor(case_sensitive=False)
scam_terms = {
    "telegram": "🚨 **CRITICAL:** 'Telegram' is used 99% by scammers.",
    "whatsapp": "🚨 **CRITICAL:** 'WhatsApp' interview request detected.",
    "signal": "🚨 **CRITICAL:** 'Signal App' is a major red flag.",
    "wire transfer": "🚨 **CRITICAL:** 'Wire Transfer' request detected.",
    "kindly deposit": "💸 **FRAUD:** 'Kindly deposit' is a known scam phrase.",
    "cashier check": "💸 **FRAUD:** 'Cashier Check' is a common banking scam.",
    "no interview": "⚠️ **Suspicious:** Skipping interview process is a scam tactic.",
    "update my profile": "🚨 **PHISHING:** Asking to click links to 'update profile'.",
}
for term, reason in scam_terms.items():
    scam_processor.add_keyword(term, reason)

# --- GREEN FLAGS (LEGITIMACY BOOSTERS) ---
GREEN_FLAGS = ["401k", "health insurance", "pto", "tuition reimbursement", "dental", "vision", "on-site"]

# ==========================================
# 2. HELPER FUNCTIONS & CLASSES
# ==========================================

# --- A. Structural Feature Extractor ---
def extract_structural_features(text):
    text = str(text)
    length = len(text) if len(text) > 0 else 1
    
    caps_count = sum(1 for c in text if c.isupper())
    digit_count = sum(1 for c in text if c.isdigit())
    special_count = len(re.findall(r'[^a-zA-Z0-9\s]', text))
    has_email = 1 if re.search(r'\S+@\S+', text) else 0
    has_phone = 1 if re.search(r'\b\d{10}\b|\+\d{1,3}', text) else 0
    has_url = 1 if "http" in text or "www" in text else 0
    word_count = len(text.split())

    # Order matters! Must match Training Script
    return [
        caps_count/length, digit_count/length, special_count/length, 
        has_email, has_phone, has_url, word_count
    ]

# --- B. CLASSES FOR SUPERVISED MODEL ---
class TextCleaner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        cleaned = []
        for text in X:
            text = str(text).lower() if text else ""
            text = REGEX_HTTP.sub('token_url', text)
            text = REGEX_EMAIL.sub('token_email', text)
            text = REGEX_SPECIAL_CHARS.sub('', text)
            text = REGEX_SPACES.sub(' ', text).strip()
            cleaned.append(text if text else "token_empty_input")
        return cleaned

class SpacyVectorTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, nlp=None): self.nlp = nlp
    def fit(self, X, y=None): return self
    def transform(self, X):
        if hasattr(g, 'spacy_doc') and g.spacy_doc is not None and len(X) == 1:
            doc = g.spacy_doc
            return np.array([doc.vector if doc.has_vector else np.zeros(300)])
        if self.nlp is None:
            if 'nlp_engine' in globals() and nlp_engine: self.nlp = nlp_engine
            else: return np.zeros((len(X), 300))
        docs = list(self.nlp.pipe(X, disable=["ner", "parser"]))
        return np.array([doc.vector if doc.has_vector else np.zeros(300) for doc in docs])

# --- C. ROBUST ANOMALY DETECTOR CLASS (UPDATED WITH DETAILED XAI) ---
class RobustAnomalyDetector(BaseEstimator, ClassifierMixin):
    def __init__(self, input_dim=307):
        self.input_dim = input_dim
        self.iso_model = None 
        self.ae_threshold = 0.0
        self.ae_weights = None 
        self.autoencoder = None 

    def _build_autoencoder(self):
        # ⚠️ Architecture matches the training script
        model = Sequential([
            Input(shape=(self.input_dim,)),
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(128, activation='relu'),
            Dense(32, activation='relu'),
            Dense(128, activation='relu'),
            Dense(256, activation='relu'),
            Dense(self.input_dim, activation='linear')
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    def predict_with_explanation(self, vector_and_features):
        if self.autoencoder is None:
            self.autoencoder = self._build_autoencoder()
            self.autoencoder.set_weights(self.ae_weights)

        input_data = vector_and_features.reshape(1, -1)
        input_scaled = self.scaler.transform(input_data)
        
        iso_pred = self.iso_model.predict(input_scaled)[0]
        recon = self.autoencoder.predict(input_scaled, verbose=0)
        loss = tf.keras.losses.mse(recon, input_scaled).numpy()[0]
        
        explanations = []

        # --- 🧠 LOGIC: CUSTOM EXPLAINER FOR ANOMALY ---
        # Features Index mapping (based on extract_structural_features):
        # The last 7 items in the 307-dim vector are our stats.
        stats = vector_and_features[300:] 
        caps_ratio = stats[0]
        digit_ratio = stats[1]
        special_ratio = stats[2]
        word_count = stats[6]

        if iso_pred == -1:
            msg = "Statistical Outlier detected."
            if word_count < 10: msg += " (Text is unusually short)."
            elif word_count > 500: msg += " (Text is unusually long)."
            
            explanations.append({
                "type": "Statistical Outlier",
                "message": msg
            })
            
        if loss > self.ae_threshold:
            reasons = []
            if caps_ratio > 0.15: reasons.append("Too many Capital Letters")
            if special_ratio > 0.05: reasons.append("Excessive Symbols (!@#)")
            if digit_ratio > 0.1: reasons.append("High usage of Digits")
            
            reason_str = ", ".join(reasons) if reasons else "Unusual text pattern"
            
            explanations.append({
                "type": "Structural Anomaly",
                "message": f"Suspicious structure detected ({reason_str})."
            })
            
        return explanations

# ==========================================
# 3. LOAD RESOURCES
# ==========================================
if __name__ != '__main__':
    sys.modules['__main__'] = sys.modules[__name__]

# A. SPACY
SPACY_AVAILABLE = False
nlp_engine = None
try:
    import spacy
    try:
        nlp_engine = spacy.load("en_core_web_lg")
        SPACY_AVAILABLE = True
        log_debug("Spacy 'lg' loaded.", "SUCCESS")
    except:
        nlp_engine = spacy.blank("en")
        log_debug("Spacy failed. Using Blank.", "ERROR")
except ImportError:
    log_debug("Spacy Not Installed.", "ERROR")

# B. MODELS
supervised_model = None
anomaly_model = None
explainer = LimeTextExplainer(class_names=['Real', 'Fake'])

SUPERVISED_FILE = 'production_fake_job_pipeline.pkl'
ANOMALY_FILE = 'robust_anomaly_model.pkl'

def inject(est):
    if isinstance(est, SpacyVectorTransformer): est.nlp = nlp_engine; return True
    if hasattr(est, 'steps'): [inject(s[1]) for s in est.steps]
    if hasattr(est, 'transformer_list'): [inject(s[1]) for s in est.transformer_list]
    if hasattr(est, 'estimator'): inject(est.estimator)

if os.path.exists(SUPERVISED_FILE):
    try:
        supervised_model = joblib.load(SUPERVISED_FILE)
        inject(supervised_model)
        log_debug("Supervised Model Loaded.", "SUCCESS")
    except Exception as e:
        log_debug(f"Supervised Load Failed: {str(e)}", "ERROR")

if os.path.exists(ANOMALY_FILE):
    try:
        anomaly_model = joblib.load(ANOMALY_FILE)
        log_debug("Unsupervised Model Loaded.", "SUCCESS")
    except Exception as e:
        log_debug(f"Anomaly Load Failed: {str(e)}", "ERROR")

# ==========================================
# 4. PREDICTION LOGIC
# ==========================================
@app.route('/predict', methods=['POST'])
def predict():
    current_user = session.get('user', '')
    is_admin = (current_user == 'Yoge')
    trace_logs = []

    def trace(msg, lvl="INFO"):
        log_debug(msg, lvl)
        if is_admin:
            timestamp = datetime.now().strftime("%H:%M:%S")
            trace_logs.append(f"[{timestamp}] [{lvl}] {msg}")

    if not current_user: return jsonify({'error': 'Unauthorized'}), 401
    
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        text_lower = text.lower()
        if not text: return jsonify({'error': 'Empty data'}), 400

        # CACHE CHECK
        text_hash = hashlib.md5(text_lower.encode('utf-8')).hexdigest()
        cached_result = cache.get(text_hash)
        if cached_result:
            trace("⚡ Cache Hit!", "SUCCESS")
            cached_result['system_logs'] = list(reversed(trace_logs))
            return jsonify(cached_result)

        # START ANALYSIS
        trace(f"📉 Analyzing: {len(text)} chars", "DEBUG")
        g.spacy_doc = None
        reasons = []
        is_gibberish = False

        # 1. HARDCODED CHECKS (FlashText & Regex)
        trace("🔍 Running Rule-Based Checks...", "DEBUG")
        found_scam_reasons = scam_processor.extract_keywords(text)
        found_scam_reasons = list(set(found_scam_reasons))
        has_scam_keywords = len(found_scam_reasons) > 0
        
        # Check for Green Flags (Benefits)
        green_hits = [g for g in GREEN_FLAGS if g in text_lower]
        has_green = len(green_hits) > 0

        # Check for Gibberish/Code
        gibberish_reason = None
        if not has_scam_keywords:
            if REGEX_CODE_SNIPPET.search(text):
                is_gibberish = True
                gibberish_reason = "🚫 **Input Error:** Code snippet or SQL syntax detected."
            elif REGEX_LONG_STRING.search(text) and "http" not in text:
                is_gibberish = True
                gibberish_reason = "🚫 **Input Error:** Suspicious long strings detected."
            elif REGEX_REPEATED_CHARS.search(text_lower):
                is_gibberish = True
                gibberish_reason = "🚫 **Gibberish:** Repetitive characters detected."
            else:
                words = text_lower.split()
                if len(words) > 10 and len(set(words)) / len(words) < 0.2:
                    is_gibberish = True
                    gibberish_reason = "🚫 **Spam:** Excessive word repetition."

        # 2. SPACY PREP
        if SPACY_AVAILABLE and nlp_engine:
            doc = nlp_engine(text)
            g.spacy_doc = doc
            if len(doc) < 5:
                is_gibberish = True
                gibberish_reason = "🚫 **Input Error:** Text too short to analyze."

        # 3. SUPERVISED PREDICTION (SMART LOGIC) 🧠
        fake_prob = 0.0
        if supervised_model:
            trace("🤖 Running Supervised Model (Smart Logic)...", "DEBUG")
            base_prob = supervised_model.predict_proba([text])[0][1]
            trace(f"   -> Base Score: {base_prob:.4f}", "DEBUG")
            
            # Sliding Window (For long text)
            max_window_prob = 0.0
            sentences = re.split(r'(?<=[.!?]) +', text)
            if len(sentences) >= 3:
                windows = [" ".join(sentences[i:i+3]) for i in range(len(sentences)-2)]
                if windows:
                    window_probs = supervised_model.predict_proba(windows)[:, 1]
                    max_window_prob = np.max(window_probs)
                    trace(f"   -> Sliding Window Max: {max_window_prob:.4f}", "DEBUG")
            
            fake_prob = max(base_prob, max_window_prob)
        else:
            trace("❌ Supervised Model Missing", "ERROR")

        # 4. UNSUPERVISED ANOMALY CHECK 🦄
        anomaly_warnings = []
        if anomaly_model:
            trace("🦄 Running Anomaly Detector...", "DEBUG")
            try:
                if hasattr(g, 'spacy_doc') and g.spacy_doc:
                    vec = g.spacy_doc.vector
                else:
                    vec = nlp_engine(text).vector
                
                stats = extract_structural_features(text)
                final_features = np.hstack((vec, np.array(stats)))
                
                anomaly_warnings = anomaly_model.predict_with_explanation(final_features)
                if anomaly_warnings:
                    trace(f"   -> {len(anomaly_warnings)} Anomalies Found!", "WARN")
            except Exception as ae:
                trace(f"Anomaly Error: {str(ae)}", "ERROR")

        # 5. MERGE & CALCULATE FINAL VERDICT ⚖️
        human_reasons = []
        override_active = False

        # --- 🔧 ANOMALY BOOST LOGIC ---
        if anomaly_warnings:
            # Boost score to Critical (85%)
            fake_prob = max(fake_prob, 0.85)
            # Add specific explanations to UI
            for warn in anomaly_warnings:
                human_reasons.append(f"⚠️ **Anomaly:** {warn['message']}")
            trace(f"🚀 Anomaly Boost Applied! Score bumped to {fake_prob:.2f}", "WARN")

        # A. Critical Triggers
        if has_scam_keywords:
            fake_prob = 0.99
            human_reasons.extend(found_scam_reasons)
            override_active = True
            trace("🔒 Critical Trigger: Override to 99%", "WARN")

        # B. Gibberish
        if is_gibberish and gibberish_reason:
            fake_prob = max(fake_prob, 0.95)
            human_reasons.append(gibberish_reason)

        # C. Green Flags
        if has_green and not override_active and not is_gibberish and not anomaly_warnings:
            original_prob = fake_prob
            fake_prob = min(fake_prob, 0.30)
            if original_prob > 0.5:
                trace(f"🛡️ Benefits found. Reduced score from {original_prob:.2f}", "SUCCESS")
                human_reasons.append("✅ **Legitimacy Boost:** Verified corporate benefits found.")

        # D. LIME & Soft Triggers
        if not override_active and not is_gibberish:
            if "@gmail.com" in text_lower or "@yahoo.com" in text_lower:
                fake_prob += 0.20
                human_reasons.append("⚠️ **Suspicious:** Personal email domain.")
            
            # --- 🔧 FIX: ALWAYS RUN LIME IF > 10% ---
            if fake_prob > 0.10:
                try:
                    trace("🍋 Running LIME...", "DEBUG")
                    exp = explainer.explain_instance(text, supervised_model.predict_proba, num_features=5, num_samples=50)
                    lime_list = exp.as_list()
                    suspicious_words = [w for w, s in lime_list if s > 0.05]
                    if suspicious_words:
                        human_reasons.append(f"🔍 **AI Insight:** Risky words: '{', '.join(suspicious_words)}'")
                except: pass

        # 6. FINAL SCORING
        fake_prob = min(max(fake_prob, 0.0), 1.0)
        
        if fake_prob > 0.8:
            if not human_reasons: human_reasons.append("🤖 **AI Verdict:** Highly suspicious pattern.")
        elif fake_prob > 0.5:
            if not human_reasons: human_reasons.append("🤖 **AI Verdict:** Text resembles known scam templates.")
        else:
            if not human_reasons and not anomaly_warnings: 
                human_reasons.append("✅ **System Clean:** No threats found.")

        trace(f"🏆 Final Score: {fake_prob:.4f}", "SUCCESS")

        result = {
            'fraud_probability': round(fake_prob * 100, 2), 
            'reasons': human_reasons, 
            'is_gibberish': is_gibberish,
            'anomaly_analysis': {
                'detected': len(anomaly_warnings) > 0,
                'warnings': anomaly_warnings
            }
        }

        # Cache & Return
        result['trace_logs'] = trace_logs
        cache.set(text_hash, result)
        result['system_logs'] = list(reversed(trace_logs)) if is_admin else None
        
        return jsonify(result)

    except Exception as e:
        log_debug(f"Internal Error: {str(e)}", "ERROR")
        return jsonify({'error': str(e)}), 500

# ==========================================
# 5. AUTH & DB ROUTING
# ==========================================
def init_db():
    with sqlite3.connect(DB_NAME) as conn:
        conn.cursor().execute('''CREATE TABLE IF NOT EXISTS users 
            (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT UNIQUE NOT NULL, email TEXT UNIQUE NOT NULL, password TEXT NOT NULL)''')
if not os.path.exists(DB_NAME): init_db()

@app.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    return response

@app.route('/api/signup', methods=['POST'])
def api_signup():
    data = request.get_json()
    hashed = generate_password_hash(data.get('password'))
    try:
        with sqlite3.connect(DB_NAME) as conn:
            conn.cursor().execute("INSERT INTO users (username, email, password) VALUES (?, ?, ?)", 
                                (data.get('username'), data.get('email'), hashed))
            session['user'] = data.get('username')
            session.permanent = True
            return jsonify({'success': True, 'username': data.get('username')})
    except: return jsonify({'error': 'User exists'}), 409

@app.route('/api/login', methods=['POST'])
def api_login():
    data = request.get_json()
    with sqlite3.connect(DB_NAME) as conn:
        row = conn.cursor().execute("SELECT password FROM users WHERE username = ?", (data.get('username'),)).fetchone()
        if row and check_password_hash(row[0], data.get('password')):
            session['user'] = data.get('username')
            session.permanent = True if data.get('remember') else False
            return jsonify({'success': True, 'username': data.get('username')})
    return jsonify({'error': 'Invalid credentials'}), 401

@app.route('/api/logout', methods=['POST'])
def api_logout(): session.clear(); return jsonify({'success': True})

@app.route('/api/delete_account', methods=['POST'])
def api_delete_account():
    with sqlite3.connect(DB_NAME) as conn:
        conn.cursor().execute("DELETE FROM users WHERE username = ?", (session.get('user'),))
    session.clear()
    return jsonify({'success': True})

@app.route('/')
def login_page():
    if 'user' in session: return redirect(url_for('dashboard_page'))
    return render_template('login.html')

@app.route('/dashboard')
def dashboard_page():
    if 'user' not in session: return redirect(url_for('login_page'))
    return render_template('index.html', username=session['user'])

if __name__ == '__main__':
    app.run(debug=True)
