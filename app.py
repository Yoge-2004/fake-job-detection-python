import warnings
from datetime import datetime
import traceback

# ==========================================
# 0. CRADLE LOGGING
# ==========================================
SERVER_LOGS = []

def log_debug(message, level="INFO"):
    timestamp = datetime.now().strftime("%H:%M:%S")
    entry = f"[{timestamp}] [{level}] {message}"
    SERVER_LOGS.append(entry)
    if len(SERVER_LOGS) > 200: SERVER_LOGS.pop(0)
    print(entry, flush=True)

def custom_warning_handler(message, category, filename, lineno, file=None, line=None):
    pass 

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

import torch
import torch.nn.functional as F
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
import __main__

# 🔧 MONKEY PATCH for NumPy compatibility
if not hasattr(np, 'object'): np.object = object
if not hasattr(np, 'bool'): np.bool = bool
if not hasattr(np, 'int'): np.int = int

log_debug("--- SYSTEM BOOT SEQUENCE INITIATED ---", "STARTUP")

# ==========================================
# 2. CLASS DEFINITIONS (REQUIRED FOR OLD PIPELINE)
# ==========================================
class TextCleaner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        cleaned = []
        for text in X:
            text = str(text).lower() if text else ""
            text = re.sub(r'\s+', ' ', text).strip()
            cleaned.append(text)
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
# 3. HEURISTIC & VALIDATION ENGINE
# ==========================================
def heuristic_analysis(text):
    text_lower = text.lower(); warnings = []
    
    behavioral_patterns = [
        (r"(validate|verify).{0,20}(bank|account|wallet)", "🎣 **Phishing:** Request to validate financial info."),
        (r"(click|follow).{0,20}(link|url).{0,20}(verify|update)", "🎣 **Phishing:** 'Click link to verify' pattern."),
        (r"(processing|training).{0,10}(fee|charge|cost)", "💸 **Financial:** Illegal demand for fees."),
        (r"(no).{0,10}(interview).{0,20}(direct)", "⚠️ **Red Flag:** Direct hire / No interview.")
    ]
    for pattern, msg in behavioral_patterns:
        if re.search(pattern, text_lower): warnings.append(msg)

    triggers = {
        "telegram": "🚨 **Platform:** Telegram contact.",
        "signal": "🚨 **Platform:** Signal (Encrypted) contact.",
        "whatsapp": "🚨 **Platform:** WhatsApp contact.",
        "usdt": "🚨 **Crypto:** USDT payment mentioned.",
        "bitcoin": "🚨 **Crypto:** Bitcoin payment mentioned.",
        "anydesk": "⚠️ **Security:** Remote Access Tool (AnyDesk)."
    }
    for word, msg in triggers.items():
        if word in text_lower: warnings.append(msg)
    return warnings

def metadata_check(text):
    advisory = []
    text_low = text.lower()
    if "@gmail.com" in text_low or "@yahoo.com" in text_low: 
        advisory.append("ℹ️ **Identity:** Personal email domain used.")
    if "salary" not in text_low and "$" not in text and "lpa" not in text_low: 
        advisory.append("ℹ️ **Clarity:** Missing salary details.")
    if "linkedin" not in text_low and "company" not in text_low: 
        advisory.append("ℹ️ **Verification:** No company/social links.")
    return advisory

def extract_structural_features(text):
    text = str(text); length = max(len(text), 1)
    caps = sum(1 for c in text if c.isupper())
    digits = sum(1 for c in text if c.isdigit())
    specials = len(re.findall(r'[^a-zA-Z0-9\s]', text))
    word_count = len(text.split())
    return [caps/length, digits/length, specials/length, 1 if "@" in text else 0, 0, 1 if "http" in text else 0, word_count]

from wordfreq import zipf_frequency

def detect_invalid_language(text):
    if not text: return False, []
    issues = []
    length = len(text)
    
    non_ascii_count = len(re.findall(r'[^\x00-\x7F]', text))
    if (non_ascii_count / length) > 0.2: 
        msg = "Language Error: Non-English text detected"
        issues.append(msg)
        return True, issues 
    
    code_symbols = len(re.findall(r'[\{\}\<\>;=\[\]]', text))
    if (code_symbols / length) > 0.10: 
        msg = "Language Error: Source Code or HTML detected"
        issues.append(msg)
        return True, issues

    code_signatures = ["def __init__", "public static void", "<script>", "SELECT * FROM"]
    if any(sig in text for sig in code_signatures):
        msg = "Language Error: Programming Code Detected"
        issues.append(msg)
        return True, issues

    tokens = [t for t in text.split() if t.isalpha()]
    if not tokens: return False, [] 
    
    unknown_word_count = 0
    total_checked = 0
    for token in tokens:
        if len(token) < 4: continue 
        total_checked += 1
        lower_token = token.lower()
        if zipf_frequency(lower_token, 'en') > 0.0: continue 
        is_compound = False
        if len(lower_token) > 6: 
            for i in range(3, len(lower_token) - 2): 
                part1 = lower_token[:i]
                part2 = lower_token[i:]
                if zipf_frequency(part1, 'en') > 0.0 and zipf_frequency(part2, 'en') > 0.0:
                    is_compound = True
                    break
        if is_compound: continue
        unknown_word_count += 1

    if total_checked > 0:
        gibberish_ratio = unknown_word_count / total_checked
        if gibberish_ratio > 0.5:
            msg = f"Language Error: Gibberish Detected ({int(gibberish_ratio*100)}% unknown)"
            issues.append(msg)
            return True, issues

    return False, []

# ==========================================
# 4. APP & MODEL LOADING
# ==========================================
app = Flask(__name__)
app.secret_key = "jobguard_production_key"
app.permanent_session_lifetime = timedelta(days=30)
DB_NAME = "users.db"
cache = Cache(app, config={"CACHE_TYPE": "SimpleCache", "CACHE_DEFAULT_TIMEOUT": 3600})

nlp_engine = None
try:
    import spacy
    nlp_engine = spacy.load("en_core_web_lg")
    log_debug("✅ Spacy Loaded", "SUCCESS")
except:
    import spacy
    nlp_engine = spacy.blank("en")
    log_debug("⚠️ Spacy Failed. Using Blank.", "WARN")

sklearn_pipeline = None
def force_inject_spacy(estimator, nlp_engine):
    if isinstance(estimator, SpacyVectorTransformer): estimator.nlp = nlp_engine
    if hasattr(estimator, 'steps'):
        for name, step in estimator.steps: force_inject_spacy(step, nlp_engine)
    if hasattr(estimator, 'transformer_list'):
        for name, trans in estimator.transformer_list: force_inject_spacy(trans, nlp_engine)

if os.path.exists('production_fake_job_pipeline.pkl'):
    try:
        sklearn_pipeline = joblib.load('production_fake_job_pipeline.pkl')
        force_inject_spacy(sklearn_pipeline, nlp_engine)
        log_debug("✅ Sklearn Pipeline Loaded", "SUCCESS")
    except Exception as e:
        log_debug(f"❌ Sklearn Load Failed: {e}", "ERROR")

bert_tokenizer = None; bert_model = None
BERT_PATH = "." 

if os.path.exists("model.safetensors"):
    try:
        bert_tokenizer = DistilBertTokenizerFast.from_pretrained(BERT_PATH)
        bert_model = DistilBertForSequenceClassification.from_pretrained(BERT_PATH)
        bert_model.eval()
        log_debug("✅ BERT Model Loaded", "SUCCESS")
    except Exception as e:
        log_debug(f"❌ BERT Load Failed: {e}", "CRITICAL")
else:
    log_debug(f"⚠️ BERT files not found in current directory.", "WARN")

anomaly_model = None
if os.path.exists('robust_anomaly_model.pkl'):
    try:
        anomaly_model = joblib.load('robust_anomaly_model.pkl')
        log_debug("✅ Anomaly Detector Loaded", "SUCCESS")
    except: pass

explainer = LimeTextExplainer(class_names=['Real', 'Fake'])

def bert_lime_predict(texts):
    if not bert_model: return np.array([[0.5, 0.5]] * len(texts))
    inputs = bert_tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        logits = bert_model(**inputs).logits
    return F.softmax(logits, dim=1).numpy()

# ==========================================
# 5. PREDICTION LOGIC (WITH FIXED OVERRIDE)
# ==========================================
@app.route('/predict', methods=['POST'])
def predict():
    trace_logs = []
    is_admin = session.get('user') == 'Yoge'
    
    def trace(msg, lvl="INFO"):
        log_debug(msg, lvl)
        if is_admin: trace_logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] [{lvl}] {msg}")

    try:
        data = request.get_json(); text = data.get('text', '').strip()
        if not text: return jsonify({'error': 'No input'}), 400
        
        # 🟢 RESTORED CACHING
        text_hash = hashlib.md5(text.lower().encode('utf-8')).hexdigest()
        cached = cache.get(text_hash)
        if cached:
            if is_admin: 
                cached['system_logs'] = [f"[CACHE] Hit for {text_hash[:8]}"] + cached.get('system_logs', [])
            return jsonify(cached)

        # 1. GIBBERISH CHECK
        is_invalid_lang, lang_issues = detect_invalid_language(text)
        if is_invalid_lang:
            response = {
                'fraud_probability': 0, 'is_gibberish': True, 'reasons': [], 
                'advisory': [], 'anomaly_analysis': lang_issues, 'xai_insights': [],
                'system_logs': [], 'verdict': "Invalid"
            }
            cache.set(text_hash, response) # Cache invalid inputs too
            return jsonify(response)

        # 2. RUN MODELS
        doc = nlp_engine(text); g.spacy_doc = doc

        # --- MODEL 1: BERT ---
        bert_score = 0.5
        if bert_model:
            inputs = bert_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad(): outputs = bert_model(**inputs)
            bert_score = F.softmax(outputs.logits, dim=1)[0][1].item()
            trace(f"BERT Confidence: {bert_score:.4f}", "AI")

        # --- MODEL 2: SKLEARN ---
        sklearn_score = 0.5
        if sklearn_pipeline:
            sklearn_score = sklearn_pipeline.predict_proba([text])[0][1]
            trace(f"Sklearn Confidence: {sklearn_score:.4f}", "AI")

        # --- MODEL 3: ANOMALY ---
        anomaly_alerts = []
        mse_value = 0.0
        if anomaly_model:
            stats = extract_structural_features(text)
            features = np.hstack((doc.vector, np.array(stats)))
            anomaly_alerts = anomaly_model.predict_with_explanation(features)
            for alert in anomaly_alerts:
                match = re.search(r"MSE:\s*([\d\.]+)", alert)
                if match: mse_value = float(match.group(1))
            
            # 🟢 FIX: IGNORE TINY MSE NOISE (Like 0.0045)
            if mse_value > 0.008: 
                trace(f"Anomaly Detected (MSE {mse_value:.4f})", "WARN")
            else:
                # Silently drop very weak anomalies
                anomaly_alerts = [] 

        # =========================================================
        # 🧠 SCORING LOGIC
        # =========================================================
        
        final_prob = 0.0
        
        # 1. BERT AUTHORITY
        if bert_score > 0.85:
            final_prob = bert_score
            trace("Logic: BERT Authority", "RESULT")
            
        # 2. SKLEARN BACKUP
        elif sklearn_score > 0.80:
            final_prob = sklearn_score
            trace("Logic: Sklearn Override", "RESULT")

        # 3. CONSENSUS
        elif bert_score > 0.60 and sklearn_score > 0.60:
            final_prob = (bert_score + sklearn_score) / 2
            trace("Logic: Moderate Consensus", "RESULT")

        # 4. REVIEW LOGIC (With Relaxed "Proven Safe" Threshold)
        else:
            max_risk = max(bert_score, sklearn_score)
            
            # 🟢 FIX: RELAXED SAFETY THRESHOLD
            # If BERT is ultra-safe (<10%), we trust it completely.
            # OR if both are reasonably safe (<30%), we trust them.
            is_proven_safe = (bert_score < 0.10) or (bert_score < 0.20 and sklearn_score < 0.30)
            
            suspects_something = False
            
            if sklearn_score > 0.45: 
                suspects_something = True
            
            if anomaly_alerts:
                if is_proven_safe:
                    trace(f"Anomaly Silenced: Overridden by Safety Logic (BERT {bert_score:.2f}, SK {sklearn_score:.2f})", "INFO")
                else:
                    suspects_something = True
                    trace("Trigger: Anomaly Validated (Models Uncertain)", "INFO")
            
            if suspects_something:
                final_prob = max(max_risk, 0.45)
                trace("Logic: Suspicion Validated -> Force Review Required", "WARN")
            else:
                final_prob = max_risk
                trace("Logic: System Clean", "RESULT")

        trace(f"Final Scoring: {final_prob:.4f}", "RESULT")

        # XAI Generation (🟢 FIX: Limit samples to 20 to prevent crash)
        lime_insights = []
        if final_prob > 0.35:
            try:
                exp = explainer.explain_instance(text, bert_lime_predict, num_features=4, num_samples=20)
                lime_insights = [f"**{w}**" for w, s in exp.as_list() if s > 0]
            except: pass

        response = {
            'fraud_probability': round(final_prob * 100, 2),
            'reasons': heuristic_analysis(text),
            'advisory': metadata_check(text),
            'anomaly_analysis': anomaly_alerts,
            'is_gibberish': False,
            'xai_insights': lime_insights,
            'system_logs': list(reversed(trace_logs)),
            'verdict': "Fake" if final_prob > 0.50 else ("Review" if final_prob > 0.35 else "Real")
        }
        
        # 🟢 RESTORED CACHING
        cache.set(text_hash, response)
        
        return jsonify(response)

    except Exception as e:
        trace(f"FATAL: {str(e)}", "ERROR")
        return jsonify({'error': str(e)}), 500
        
# ==========================================
# 6. DATABASE & AUTH ROUTES
# ==========================================
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
            session['user'] = data['username']
            session.permanent = data.get('remember', False)
            return jsonify({'success': True, 'username': data['username']})
    return jsonify({'error': 'Invalid credentials'}), 401

@app.route('/api/logout', methods=['POST'])
def api_logout(): session.clear(); return jsonify({'success': True})

@app.route('/api/delete_account', methods=['POST'])
def delete_account():
    if 'user' not in session: return jsonify({'error': 'Unauthorized'}), 401
    try:
        username = session['user']
        with sqlite3.connect(DB_NAME) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM users WHERE username = ?", (username,))
            conn.commit()
        session.clear()
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/user_info')
def user_info(): return jsonify({'username': session.get('user')})

@app.route('/api/system_logs')
def get_logs():
    if session.get('user') != 'Yoge': return jsonify([])
    return jsonify(list(reversed(SERVER_LOGS)))

@app.route('/')
def home(): return render_template('login.html')

@app.route('/dashboard')
def dashboard():
    if 'user' not in session: return redirect(url_for('home'))
    return render_template('index.html', username=session['user'])

if __name__ == '__main__':
    app.run(debug=True)
