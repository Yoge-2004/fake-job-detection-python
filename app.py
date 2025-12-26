import warnings
# warnings.filterwarnings("ignore") 

from flask import Flask, render_template, request, jsonify, session, redirect, url_for, g
from flask_caching import Cache
from sklearn.base import BaseEstimator, TransformerMixin
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

app = Flask(__name__)
app.secret_key = "jobguard_super_secret_key"
app.permanent_session_lifetime = timedelta(days=30) 
DB_NAME = "users.db"

# ==========================================
# 0. CACHE CONFIGURATION
# ==========================================
cache_config = {
    "CACHE_TYPE": "SimpleCache", 
    "CACHE_DEFAULT_TIMEOUT": 3600
}
app.config.from_mapping(cache_config)
cache = Cache(app)

# ==========================================
# 1. OPTIMIZATIONS & FLASHTEXT
# ==========================================
REGEX_LONG_STRING = re.compile(r'\S{40,}')
REGEX_REPEATED_CHARS = re.compile(r'(.)\1{4,}')
REGEX_CONSONANT_SMASH = re.compile(r'[aeiouy]')
REGEX_HTTP = re.compile(r'http\S+|www\.\S+')
REGEX_EMAIL = re.compile(r'\S+@\S+')
REGEX_SPECIAL_CHARS = re.compile(r'[^a-z0-9\s\$\%\@\.\,\!]')
REGEX_SPACES = re.compile(r'\s+')
REGEX_WORD_TOKEN = re.compile(r'\w+')

# Global Server Logs (For Console/History)
SERVER_LOGS = []
explainer = LimeTextExplainer(class_names=['Real', 'Fake'])

# --- FLASHTEXT SETUP ---
scam_processor = KeywordProcessor(case_sensitive=False)
scam_terms = {
    "telegram": "🚨 **CRITICAL:** 'Telegram' is used 99% by scammers.",
    "whatsapp": "🚨 **CRITICAL:** 'WhatsApp' interview request detected.",
    "signal": "🚨 **CRITICAL:** 'Signal App' is a major red flag for interviews.",
    "wire": "🚨 **CRITICAL:** 'Wire Transfer' request detected.",
    "kindly deposit": "💸 **FRAUD:** 'Kindly deposit' is a known scam phrase.",
    "check to purchase": "💸 **FRAUD:** Fake Equipment Check Scam detected.",
    "send a check": "💸 **FRAUD:** Fake Check Scam detected.",
    "cashier check": "💸 **FRAUD:** 'Cashier Check' is a common banking scam.",
    "no human resources": "⚠️ **Suspicious:** 'No HR screening' is highly unusual.",
    "no interview": "⚠️ **Suspicious:** Skipping interview process is a scam tactic.",
    "encrypted credential": "⚠️ **Suspicious:** Asking for credentials via chat is unsafe."
}
for term, reason in scam_terms.items():
    scam_processor.add_keyword(term, reason)

def log_debug(message, level="INFO"):
    timestamp = datetime.now().strftime("%H:%M:%S")
    entry = f"[{timestamp}] {level}: {message}"
    SERVER_LOGS.append(entry)
    if len(SERVER_LOGS) > 100: SERVER_LOGS.pop(0)
    print(entry, flush=True)

# ==========================================
# 2. CUSTOM CLASSES
# ==========================================
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

if __name__ != '__main__':
    sys.modules['__main__'] = sys.modules[__name__]

SPACY_AVAILABLE = False
nlp_engine = None

try:
    import spacy
    try:
        nlp_engine = spacy.load("en_core_web_lg")
        SPACY_AVAILABLE = True
        log_debug("Spacy 'lg' loaded.", "SUCCESS")
    except:
        try:
            nlp_engine = spacy.load("en_core_web_md")
            SPACY_AVAILABLE = True
            log_debug("Spacy 'lg' missing. Loaded 'md'.", "WARN")
        except:
            nlp_engine = spacy.blank("en")
            log_debug("Spacy failed. Using Blank.", "ERROR")
except ImportError:
    log_debug("Spacy Not Installed.", "ERROR")

model = None
MODEL_FILE = 'production_fake_job_pipeline.pkl'
if os.path.exists(MODEL_FILE):
    try:
        model = joblib.load(MODEL_FILE)
        def inject(est):
            if isinstance(est, SpacyVectorTransformer): est.nlp = nlp_engine; return True
            if hasattr(est, 'steps'): [inject(s[1]) for s in est.steps]
            if hasattr(est, 'transformer_list'): [inject(s[1]) for s in est.transformer_list]
            if hasattr(est, 'estimator'): inject(est.estimator)
        inject(model)
        log_debug("AI Model Loaded.", "SUCCESS")
    except Exception as e:
        log_debug(f"Model Load Failed: {str(e)}", "ERROR")

# ==========================================
# 3. PREDICTION LOGIC (VERBOSE LOGGING)
# ==========================================
@app.route('/predict', methods=['POST'])
def predict():
    current_user = session.get('user', '')
    is_admin = (current_user == 'Yoge')
    
    # Trace Logs: To be sent to UI (Specific to this request)
    trace_logs = [] 
    
    def trace(msg, lvl="INFO"):
        """Adds log to both Server Console and UI Response"""
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

        # --- CACHE CHECK ---
        text_hash = hashlib.md5(text_lower.encode('utf-8')).hexdigest()
        cached_result = cache.get(text_hash)
        
        if cached_result:
            trace("⚡ Cache Hit! Retrieving stored analysis...", "SUCCESS")
            # Retrieve OLD logs from cache to show them again
            if 'trace_logs' in cached_result and is_admin:
                trace_logs.extend(cached_result['trace_logs'])
                cached_result['system_logs'] = list(reversed(trace_logs)) # Update with new hit log
            return jsonify(cached_result)

        # --- IF NOT IN CACHE, START FRESH ANALYSIS ---
        trace(f"📉 Incoming Request: {len(text)} chars", "DEBUG")
        
        g.spacy_doc = None
        is_gibberish = False
        reasons = []

        # A. FAST SCAM CHECK (FlashText)
        trace("🔍 Running FlashText Keyword Search...", "DEBUG")
        found_scam_reasons = scam_processor.extract_keywords(text)
        found_scam_reasons = list(set(found_scam_reasons))
        
        has_scam_keywords = len(found_scam_reasons) > 0
        if has_scam_keywords:
            trace(f"🚨 FlashText Found Triggers: {len(found_scam_reasons)} detected", "WARN")
        else:
            trace("✅ FlashText: No Hardcoded Scams found.", "DEBUG")

        # B. REGEX CHECKS
        if not has_scam_keywords:
            trace("🧩 Running Regex Pattern Matching...", "DEBUG")
            if REGEX_LONG_STRING.search(text) and "http" not in text:
                is_gibberish = True
                trace("🚫 Regex: Suspicious Long String detected.", "WARN")
                reasons.append("🚫 **Input Error:** Suspicious long strings detected.")
            
            if REGEX_REPEATED_CHARS.search(text_lower):
                is_gibberish = True
                trace("🚫 Regex: Repetitive Characters detected.", "WARN")
                reasons.append("🚫 **Gibberish:** Repetitive characters detected.")

            if not is_gibberish:
                words_raw = text_lower.split()
                for w in words_raw:
                    if len(w) > 10 and not REGEX_CONSONANT_SMASH.search(w) and "http" not in w:
                        if not any(char.isdigit() for char in w): 
                            is_gibberish = True
                            trace(f"🚫 Regex: Consonant Smash in '{w}'", "WARN")
                            reasons.append("🚫 **Gibberish:** Random key-mashing detected.")
                            break

        # C. SPACY CHECK
        if not is_gibberish and not has_scam_keywords:
            trace(f"🧠 Spacy ({nlp_engine.meta['name'] if nlp_engine else 'None'}) Analysis Started...", "DEBUG")
            total_words = 0
            valid_words = 0
            if SPACY_AVAILABLE and nlp_engine:
                doc = nlp_engine(text)
                g.spacy_doc = doc 
                for t in doc:
                    if t.is_alpha or t.like_num or t.is_currency or not t.is_punct:
                        if not t.is_space and not t.is_punct:
                            total_words += 1
                            if t.has_vector or t.like_num or t.is_currency:
                                valid_words += 1
            else:
                words = REGEX_WORD_TOKEN.findall(text_lower)
                total_words = len(words)
                valid_words = total_words

            ratio = valid_words / total_words if total_words > 0 else 0
            trace(f"📊 Spacy Stats: {valid_words}/{total_words} Valid Words (Ratio: {ratio:.2f})", "INFO")
            
            if total_words > 0 and total_words < 5 and ratio < 0.75:
                is_gibberish = True
                trace("🚫 Spacy: Text too short.", "WARN")
                reasons.append("🚫 **Unknown Data:** Short text must be valid English.")
            elif 5 <= total_words <= 20 and ratio < 0.4:
                is_gibberish = True
                trace("🚫 Spacy: Low valid word ratio (< 0.4).", "WARN")
                reasons.append("🚫 **Gibberish:** Text contains mostly random words.")
            elif total_words > 20 and ratio < 0.15: 
                is_gibberish = True
                trace("🚫 Spacy: Very low valid word ratio (< 0.15).", "WARN")
                reasons.append("🚫 **Gibberish:** Text structure is incoherent.")

        # D. PREDICTION
        result = {}
        if is_gibberish:
            trace("❌ Verdict: Gibberish Detected.", "ERROR")
            result = {'fraud_probability': 100.0, 'reasons': reasons, 'is_gibberish': True}
        elif model:
            # 1. Base AI Prediction
            trace("🤖 Invoking Scikit-Learn Pipeline...", "DEBUG")
            proba = model.predict_proba([text])[0]
            fake_prob = proba[1]
            trace(f"🔢 Base Model Probability: {fake_prob:.4f}", "INFO")
            
            human_reasons = []
            override_active = False

            # 2. CRITICAL TRIGGERS (FlashText)
            if has_scam_keywords:
                fake_prob = 0.99
                human_reasons.extend(found_scam_reasons)
                override_active = True
                trace("🔒 Hardcoded Override Active: Score set to 99%", "WARN")

            # 3. Soft Triggers & LIME
            if not override_active:
                if "@gmail.com" in text_lower or "@yahoo.com" in text_lower:
                    fake_prob += 0.30 
                    trace("⚠️ Trigger: Personal Email Domain found.", "WARN")
                    human_reasons.append("⚠️ **Suspicious:** Using personal email for corporate role.")
                if "urgent" in text_lower or "immediate" in text_lower:
                    fake_prob += 0.10
                    trace("⚠️ Trigger: Urgency keywords found.", "WARN")
                    human_reasons.append("⚠️ **Urgency:** Scammers often create fake urgency.")

                try:
                    trace("🍋 Running LIME Explainer (Samples=500)...", "DEBUG")
                    exp = explainer.explain_instance(text, model.predict_proba, num_features=5, num_samples=500)
                    lime_list = exp.as_list()
                    
                    suspicious_words = [w for w, s in lime_list if s > 0.05]
                    safe_words = [w for w, s in lime_list if s < -0.05]
                    
                    trace(f"🔍 LIME Suspicious: {suspicious_words}", "DEBUG")
                    trace(f"🛡️ LIME Safe: {safe_words}", "DEBUG")

                    if suspicious_words and fake_prob > 0.5:
                        human_reasons.append(f"🔍 **AI Insight:** High risk words found: '{', '.join(suspicious_words)}'")
                except Exception as lime_e:
                    trace(f"LIME Error: {str(lime_e)}", "ERROR")

            fake_prob = min(max(fake_prob, 0.0), 1.0)

            if fake_prob > 0.8:
                if not human_reasons: human_reasons.append("🤖 **AI Verdict:** Highly suspicious pattern detected.")
            elif fake_prob > 0.5:
                if not human_reasons: human_reasons.append("🤖 **AI Verdict:** Text resembles known scam templates.")
            else:
                if not human_reasons: human_reasons.append("✅ **System Clean:** No known threats detected.")

            trace(f"🏆 Final Calculated Risk Score: {fake_prob:.4f}", "SUCCESS")
            result = {'fraud_probability': round(fake_prob * 100, 2), 'reasons': human_reasons, 'is_gibberish': False}
            
            # --- SAVE TO CACHE (INCLUDE LOGS) ---
            # We save the trace_logs INSIDE the cache value so they can be replayed
            result['trace_logs'] = trace_logs 
            cache.set(text_hash, result)
            trace("💾 Result & Logs saved to Cache.", "DEBUG")

        else:
            result = {'fraud_probability': 0, 'reasons': ["Mock Mode"], 'is_gibberish': False}

        # Send logs to UI
        result['system_logs'] = list(reversed(trace_logs)) if is_admin else None
        return jsonify(result)

    except Exception as e:
        error_msg = f"Internal Error: {str(e)}"
        log_debug(error_msg, "ERROR")
        return jsonify({'error': error_msg}), 500

# ==========================================
# 4. AUTH & DB (UNCHANGED)
# ==========================================
def init_db():
    with sqlite3.connect(DB_NAME) as conn:
        conn.cursor().execute('''CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL, password TEXT NOT NULL)''')
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
        if data.get('remember'): session.permanent = True
        else: session.permanent = False
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
