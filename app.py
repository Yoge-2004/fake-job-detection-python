import warnings
from datetime import datetime

# ==========================================
# 0. CRADLE LOGGING (FIRST)
# ==========================================
SERVER_LOGS = []

def log_debug(message, level="INFO"):
    timestamp = datetime.now().strftime("%H:%M:%S")
    entry = f"[{timestamp}] [{level}] {message}"
    SERVER_LOGS.append(entry)
    if len(SERVER_LOGS) > 200: SERVER_LOGS.pop(0)
    print(entry, flush=True)

def custom_warning_handler(message, category, filename, lineno, file=None, line=None):
    log_debug(f"OS_WARNING: {message} ({category.__name__})", "WARN")

warnings.showwarning = custom_warning_handler

# ==========================================
# 1. IMPORTS & SETUP
# ==========================================
from flask import Flask, render_template, request, jsonify, session, redirect, url_for, g
from flask_caching import Cache
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.preprocessing import MinMaxScaler
import joblib, os, re, numpy as np, sqlite3, sys, hashlib, math
from datetime import timedelta
from werkzeug.security import generate_password_hash, check_password_hash
from lime.lime_text import LimeTextExplainer
from flashtext import KeywordProcessor

# 🔧 MONKEY PATCH for NumPy (Fixes FutureWarning/AttributeError)
if not hasattr(np, 'object'):
    np.object = object
if not hasattr(np, 'bool'):
    np.bool = bool
if not hasattr(np, 'int'):
    np.int = int

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
import __main__

log_debug("--- SYSTEM BOOT SEQUENCE INITIATED ---", "STARTUP")

# ==========================================
# 2. CLASS DEFINITIONS
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
            cleaned.append(re.sub(r'\s+', ' ', text).strip())
        return cleaned

class SpacyVectorTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, nlp=None): self.nlp = nlp
    def fit(self, X, y=None): return self
    def transform(self, X):
        engine = self.nlp if self.nlp else nlp_engine
        if hasattr(g, 'spacy_doc') and g.spacy_doc is not None and len(X) == 1:
            return np.array([g.spacy_doc.vector if g.spacy_doc.has_vector else np.zeros(300)])
        return np.array([doc.vector if doc.has_vector else np.zeros(300) for doc in engine.pipe(X)])

class RobustAnomalyDetector(BaseEstimator, ClassifierMixin):
    def __init__(self, input_dim=307):
        self.input_dim = input_dim
        self.scaler = MinMaxScaler()
        self.iso_model = None; self.ae_threshold = 0.0; self.ae_weights = None; self.autoencoder = None
    
    def _build_autoencoder(self):
        model = Sequential([
            Input(shape=(self.input_dim,)),
            Dense(256, activation='relu'), BatchNormalization(), Dropout(0.3),
            Dense(128, activation='relu'), Dense(32, activation='relu'),
            Dense(128, activation='relu'), Dense(256, activation='relu'),
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
        if iso_pred == -1: explanations.append("Statistical Structural Outlier")
        if loss > self.ae_threshold: explanations.append(f"Deep Pattern Anomaly (MSE: {loss:.4f})")
        return explanations

__main__.TextCleaner = TextCleaner
__main__.SpacyVectorTransformer = SpacyVectorTransformer
__main__.RobustAnomalyDetector = RobustAnomalyDetector

# ==========================================
# 3. LOGIC ENGINES (PHISHING + MONEY MULE + URGENCY)
# ==========================================
GREEN_FLAGS = ["401k", "health insurance", "pto", "dental", "vision", "on-site", "equal opportunity", "veteran status", "disability"]

def extract_structural_features(text):
    text = str(text); length = max(len(text), 1)
    caps = sum(1 for c in text if c.isupper())
    digits = sum(1 for c in text if c.isdigit())
    specials = len(re.findall(r'[^a-zA-Z0-9\s]', text))
    word_count = len(text.split())
    return [caps/length, digits/length, specials/length, 1 if "@" in text else 0, 0, 1 if "http" in text else 0, word_count]

def human_reasoning_engine(text):
    text_lower = text.lower(); reasons = []
    
    # 1. Logic Gaps
    if ("usa" in text_lower or "united states" in text_lower) and ("lpa" in text_lower or "rupees" in text_lower):
        reasons.append("⚠️ **Logic Gap:** USA location but salary in INR (LPA).")
    if "softwares" in text_lower:
        reasons.append("✍️ **Grammar Alert:** Non-standard plural 'softwares' detected.")

    # 2. Scam Phrasing
    if "kindly" in text_lower and ("pay" in text_lower or "deposit" in text_lower):
        reasons.append("💸 **Scam Phrasing:** 'Kindly pay' pattern detected.")
    if "provide" in text_lower and ("laptop" in text_lower or "equipment" in text_lower) and "work from home" in text_lower:
        reasons.append("⚠️ **Equipment Promise:** Promise of free laptop/equipment is a common scam tactic.")
        
    # 3. PHISHING & URGENCY
    if ("update" in text_lower or "fill" in text_lower) and "profile" in text_lower and ("link" in text_lower or "click" in text_lower or "button" in text_lower):
        reasons.append("🎣 **Phishing Alert:** Request to click a link to 'update profile' is a common data harvesting tactic.")
    
    if "urgent" in text_lower and ("immediate" in text_lower or "seconds" in text_lower or "now" in text_lower):
        reasons.append("⚠️ **Artificial Urgency:** Scammers create pressure ('urgent', 'do it now') to force mistakes.")

    # 4. MONEY MULE (CRITICAL FIX INCLUDED) 📦
    # Detects: "Receive packages", "Residential address", "Repackage"
    mule_keywords = ["package", "parcel", "shipment"]
    action_keywords = ["receive", "repackage", "label", "forward"]
    location_keywords = ["residential", "home address", "your address"]
    
    has_mule_action = any(a in text_lower for a in action_keywords) and any(p in text_lower for p in mule_keywords)
    has_mule_loc = any(l in text_lower for l in location_keywords)
    
    if has_mule_action and has_mule_loc:
         reasons.append("📦 **Money Mule Alert:** Job involves receiving/forwarding packages at a personal address.")

    # 5. Triggers
    triggers = {
        "telegram": "🚨 **Platform Risk:** Telegram contact request.",
        "whatsapp": "🚨 **Platform Risk:** WhatsApp contact request.",
        "wire transfer": "💸 **Financial Risk:** Wire Transfer request.",
        "training fee": "💸 **Financial Risk:** Training fee request."
    }
    for word, reason in triggers.items():
        if word in text_lower: reasons.append(reason)
        
    return reasons

def metadata_check(text):
    advisory = []
    text_low = text.lower()
    if "@gmail.com" in text_low or "@yahoo.com" in text_low: advisory.append("ℹ️ **Identity:** Personal email domain used.")
    if "salary" not in text_low and "$" not in text and "lpa" not in text_low: advisory.append("ℹ️ **Clarity:** Missing salary details.")
    if "linkedin" not in text_low and "company" not in text_low: advisory.append("ℹ️ **Verification:** No company/social links.")
    return advisory

# ==========================================
# 4. APP & RESOURCES
# ==========================================
app = Flask(__name__)
app.secret_key = "jobguard_super_secret_key"
app.permanent_session_lifetime = timedelta(days=30)
DB_NAME = "users.db"
cache = Cache(app, config={"CACHE_TYPE": "SimpleCache", "CACHE_DEFAULT_TIMEOUT": 3600})

nlp_engine = None
supervised_model = None
anomaly_model = None
explainer = LimeTextExplainer(class_names=['Real', 'Fake'])

try:
    import spacy
    nlp_engine = spacy.load("en_core_web_lg")
    log_debug(f"Spacy Model Loaded: {nlp_engine.meta['name']} (v{nlp_engine.meta['version']})", "SUCCESS")
except:
    log_debug("Spacy Load Failed, using blank", "ERROR")
    import spacy
    nlp_engine = spacy.blank("en")

# ⚡ ROBUST INJECTION (Ensures pipeline actually gets the model)
def inject(est):
    if isinstance(est, SpacyVectorTransformer): 
        est.nlp = nlp_engine
        return True
    
    injected = False
    if hasattr(est, 'steps'): 
        for s in est.steps: 
            if inject(s[1]): injected = True
    if hasattr(est, 'transformer_list'):
        for s in est.transformer_list: 
            if inject(s[1]): injected = True
    if hasattr(est, 'transformers'): # ColumnTransformer
        for s in est.transformers: 
            if inject(s[1]): injected = True
    if hasattr(est, 'estimator'):
        if inject(est.estimator): injected = True
        
    return injected

if os.path.exists('production_fake_job_pipeline.pkl'):
    try:
        supervised_model = joblib.load('production_fake_job_pipeline.pkl')
        was_injected = inject(supervised_model)
        if was_injected:
            log_debug("Supervised Pipeline Loaded & Spacy Vectors LINKED ✅", "SUCCESS")
        else:
            log_debug("Supervised Pipeline Loaded but Spacy Link FAILED ❌", "ERROR")
    except Exception as e: log_debug(f"Supervised Model Error: {e}", "CRITICAL")

if os.path.exists('robust_anomaly_model.pkl'):
    try:
        anomaly_model = joblib.load('robust_anomaly_model.pkl')
        log_debug("Anomaly Detector Loaded", "SUCCESS")
    except Exception as e: log_debug(f"Anomaly Model Error: {e}", "CRITICAL")

# ==========================================
# 5. PREDICT ROUTE
# ==========================================
@app.route('/predict', methods=['POST'])
def predict():
    is_admin = (session.get('user') == 'Yoge')
    trace_logs = []
    def trace(msg, lvl="INFO"):
        log_debug(msg, lvl)
        if is_admin: trace_logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] [{lvl}] {msg}")

    try:
        data = request.get_json(); text = data.get('text', '').strip()
        if not text: return jsonify({'error': 'No input'}), 400

        # CACHE CHECK
        text_hash = hashlib.md5(text.lower().encode('utf-8')).hexdigest()
        cached_result = cache.get(text_hash)
        if cached_result:
            if is_admin:
                cached_logs = cached_result.get('system_logs', [])
                cached_logs.insert(0, f"[{datetime.now().strftime('%H:%M:%S')}] [SUCCESS] ⚡ Cache Hit!")
                cached_result['system_logs'] = cached_logs
            return jsonify(cached_result)

        trace(f"SCAN STARTED: {len(text)} chars", "INIT")
        
        # 3. GATEKEEPER (With Bypass)
        doc = nlp_engine(text); g.spacy_doc = doc
        alpha = [t for t in doc if t.is_alpha]
        valid_ratio = len([t for t in alpha if t.has_vector]) / len(alpha) if alpha else 0
        
        # Bypass for short scams
        triggers = ["pay", "fee", "urgent", "immediately", "money", "deposit"]
        has_trigger = any(t in text.lower() for t in triggers)
        
        if not has_trigger and (valid_ratio < 0.4 or len(alpha) < 5):
            trace(f"Gatekeeper Block. Validity: {valid_ratio:.2f}", "BLOCK")
            return jsonify({'fraud_probability': 15.0, 'is_gibberish': True, 'reasons': ["🚫 **Language Error:** Text is mostly unintelligible."], 'system_logs': trace_logs})

        smash_pattern = re.compile(r'[bcdfghjklmnpqrstvwxyz]{6,}', re.IGNORECASE)
        smashes = smash_pattern.findall(text)
        if smashes:
            trace(f"Embedded Gibberish Detected: {smashes}", "BLOCK")
            return jsonify({'fraud_probability': 50.0, 'is_gibberish': True, 'reasons': [f"🚫 **Gibberish Detected:** Random sequences '{smashes[0]}'."], 'system_logs': trace_logs})

        # 4. SUPERVISED AI
        base_prob = supervised_model.predict_proba([text])[0][1]
        
        sentences = re.split(r'(?<=[.!?]) +', text)
        max_window_prob = 0.0
        if len(sentences) >= 3:
            windows = [" ".join(sentences[i:i+3]) for i in range(len(sentences)-2)]
            window_probs = supervised_model.predict_proba(windows)[:, 1]
            max_window_prob = np.max(window_probs)
            trace(f"Sliding Window Peak: {max_window_prob:.4f}", "AI")
        
        final_prob = max(base_prob, max_window_prob)
        trace(f"Base AI Score: {final_prob:.4f}", "AI")

        # 5. ANOMALY DETECTOR
        anomaly_reasons = []
        if anomaly_model:
            stats = extract_structural_features(text)
            features = np.hstack((doc.vector, np.array(stats)))
            anomaly_reasons = anomaly_model.predict_with_explanation(features)
            if anomaly_reasons:
                boost = 0.10 if final_prob < 0.30 else 0.20
                final_prob = min(final_prob + boost, 0.98)
                trace(f"Anomaly Boosted. Final: {final_prob:.4f}", "WARN")

        # 6. HUMAN ENGINE
        human_reasons = human_reasoning_engine(text)
        advisory_notes = metadata_check(text)
        
        if human_reasons:
            boost = 0.25
            if any("Phishing" in r for r in human_reasons): boost = 0.40 # Critical Phishing Boost
            if any("Money Mule" in r for r in human_reasons): boost = 0.40 # Critical Mule Boost
            final_prob = min(final_prob + boost, 0.98)
            trace(f"Human Triggers Applied (+{boost})", "WARN")

        has_green = any(gf in text.lower() for gf in GREEN_FLAGS)
        if has_green and not human_reasons:
            original = final_prob
            final_prob = max(0.05, final_prob - 0.25)
            trace(f"Green Flags Found. Reduced {original:.2f} -> {final_prob:.2f}", "SUCCESS")

        # 7. XAI
        lime_insights = []
        if final_prob > 0.10:
            exp = explainer.explain_instance(text, supervised_model.predict_proba, num_features=3, num_samples=50)
            lime_insights = [f"**{w}** ({s:+.2f})" for w, s in exp.as_list() if s > 0.05]

        trace(f"SCAN COMPLETE. Verdict: {final_prob:.2%}", "SUCCESS")
        
        result = {
            'fraud_probability': round(final_prob * 100, 2),
            'reasons': human_reasons,
            'advisory': advisory_notes,
            'anomaly_analysis': anomaly_reasons,
            'xai_insights': lime_insights,
            'is_gibberish': False,
            'system_logs': list(reversed(trace_logs))
        }

        cache.set(text_hash, result)
        return jsonify(result)

    except Exception as e:
        trace(f"RUNTIME ERROR: {str(e)}", "ERROR")
        return jsonify({'error': str(e)}), 500

# ==========================================
# 6. AUTH & ROUTES
# ==========================================
@app.route('/api/user_info')
def get_user_info():
    if 'user' in session: return jsonify({'username': session['user']})
    return jsonify({'username': None})

@app.route('/api/system_logs')
def get_global_logs():
    if session.get('user') != 'Yoge': return jsonify([])
    return jsonify(list(reversed(SERVER_LOGS)))

def init_db():
    with sqlite3.connect(DB_NAME) as conn:
        conn.cursor().execute('CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT UNIQUE NOT NULL, email TEXT UNIQUE NOT NULL, password TEXT NOT NULL)')
init_db()

@app.route('/api/signup', methods=['POST'])
def api_signup():
    data = request.get_json()
    try:
        with sqlite3.connect(DB_NAME) as conn:
            conn.cursor().execute("INSERT INTO users (username, email, password) VALUES (?, ?, ?)", (data['username'], data['email'], generate_password_hash(data['password'])))
            session['user'] = data['username']; return jsonify({'success': True})
    except: return jsonify({'error': 'User exists'}), 409

@app.route('/api/login', methods=['POST'])
def api_login():
    data = request.get_json()
    with sqlite3.connect(DB_NAME) as conn:
        row = conn.cursor().execute("SELECT password FROM users WHERE username = ?", (data['username'],)).fetchone()
        if row and check_password_hash(row[0], data['password']):
            session['user'] = data['username']; return jsonify({'success': True})
    return jsonify({'error': 'Failed'}), 401

@app.route('/api/logout', methods=['POST'])
def api_logout(): session.clear(); return jsonify({'success': True})

@app.route('/api/delete_account', methods=['POST'])
def api_delete_account():
    with sqlite3.connect(DB_NAME) as conn:
        conn.cursor().execute("DELETE FROM users WHERE username = ?", (session.get('user'),))
    session.clear(); return jsonify({'success': True})

@app.route('/')
def login_page(): return render_template('login.html')

@app.route('/dashboard')
def dashboard_page():
    # 🔒 SECURITY FIX: Redirect if not logged in
    if 'user' not in session:
        return redirect(url_for('login_page'))
    return render_template('index.html', username=session['user'])

if __name__ == '__main__':
    app.run(debug=True)
