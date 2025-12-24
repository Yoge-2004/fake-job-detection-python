import warnings
# 1. SUPPRESS WARNINGS (Must be at the top)
warnings.filterwarnings("ignore", category=UserWarning)
from sklearn.exceptions import InconsistentVersionWarning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

from flask import Flask, render_template, request, jsonify, session, redirect, url_for, make_response
from sklearn.base import BaseEstimator, TransformerMixin
import joblib
import os
import random
import string
import sqlite3
import re
import numpy as np
from datetime import timedelta
from werkzeug.security import generate_password_hash, check_password_hash

# ==========================================
# 0. SPACY & LIME SAFETY IMPORT
# ==========================================
SPACY_AVAILABLE = False
LIME_AVAILABLE = False
nlp_engine = None

try:
    import spacy
    SPACY_AVAILABLE = True
    print(">> SYSTEM: Spacy library found.")
except ImportError:
    print(">> WARNING: Spacy not installed.")

try:
    from lime.lime_text import LimeTextExplainer
    LIME_AVAILABLE = True
except ImportError:
    print(">> WARNING: LIME not installed. Explanations disabled.")

app = Flask(__name__)
app.secret_key = "jobguard_super_secret_key"
app.permanent_session_lifetime = timedelta(days=7)
DB_NAME = "users.db"

# ==========================================
# 1. CUSTOM CLASSES (Must match test.py)
# ==========================================

class TextCleaner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

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

    def fit(self, X, y=None): 
        return self

    def transform(self, X):
        # Safety: If NLP is missing (e.g. mobile), use blank or zeros
        if self.nlp is None:
            if SPACY_AVAILABLE:
                try:
                    self.nlp = spacy.load("en_core_web_md")
                except:
                    self.nlp = spacy.blank("en")
            else:
                return np.zeros((len(X), 300))
        
        try:
            docs = list(self.nlp.pipe(X, disable=["ner", "parser"]))
            return np.array([doc.vector if doc.has_vector else np.zeros(300) for doc in docs])
        except Exception:
            return np.zeros((len(X), 300))

# ==========================================
# 🛑 FIX FOR HUGGING FACE DEPLOYMENT
# ==========================================
import sys
# Trick pickle into finding the classes in '__main__' even when running in Gunicorn
if __name__ != '__main__':
    sys.modules['__main__'] = sys.modules[__name__]
# ==========================================

# ==========================================
# 2. HUMAN REASONING ENGINE
# ==========================================
def explain_like_a_human(text, label, lime_features):
    reasons = []
    text_lower = text.lower()

    # 1. LOGIC GAPS
    if "usa" in text_lower or "united states" in text_lower or "california" in text_lower:
        if "lpa" in text_lower or "rupees" in text_lower:
            reasons.append("⚠️ **Geography Mismatch:** Location is USA/Global, but salary is in Indian Currency (LPA).")

    if "softwares" in text_lower:
        reasons.append("⚠️ **Grammar Red Flag:** Uses 'softwares' (incorrect plural).")

    if "kindly" in text_lower and "pay" in text_lower:
        reasons.append("⚠️ **Scam Phrasing:** 'Kindly pay' is a common phrase used in payment fraud.")

    # 2. DANGEROUS KEYWORDS
    triggers = {
        "telegram": "🚨 **Off-Platform Risk:** Asks to move chat to Telegram.",
        "whatsapp": "🚨 **Off-Platform Risk:** Asks to move chat to WhatsApp.",
        "usdt": "💸 **Crypto Risk:** Mentions USDT/Crypto payments.",
        "check": "💸 **Payment Risk:** Mentions sending a 'Check'.",
        "training fee": "💸 **Upfront Cost:** Asks for money/fees before hiring.",
        "gmail.com": "⚠️ **Generic Email:** Uses a public domain (@gmail) instead of corporate email."
    }

    for word, reason in triggers.items():
        if word in text_lower:
            reasons.append(reason)

    # 3. AI EXPLANATION
    if not reasons and label == "FAKE" and lime_features:
        top_words = [w[0] for w in lime_features if w[1] > 0][:3]
        if top_words:
            reasons.append(f"🤖 **AI Pattern Match:** Suspicious clustering around words: {', '.join(top_words)}.")

    if label == "REAL" and not reasons:
        reasons.append("✅ **Corporate Standard:** Contains professional terminology.")
        reasons.append("✅ **No Triggers:** No high-risk fraud keywords detected.")

    return reasons

# ==========================================
# 3. INTELLIGENT MODEL LOADING
# ==========================================
model = None

def load_ai_engine():
    global model, nlp_engine
    
    # 1. Load Spacy Engine
    if SPACY_AVAILABLE:
        try:
            nlp_engine = spacy.load('en_core_web_md')
        except:
            try:
                from spacy.cli import download
                download('en_core_web_md')
                nlp_engine = spacy.load('en_core_web_md')
            except:
                nlp_engine = spacy.blank("en") 

    # 2. Load Pipeline
    path = 'production_fake_job_pipeline.pkl'
    if os.path.exists(path):
        try:
            model = joblib.load(path)
            
            # 3. INJECT NLP
            def inject(est):
                if isinstance(est, SpacyVectorTransformer): est.nlp = nlp_engine; return True
                if hasattr(est, 'steps'): [inject(s[1]) for s in est.steps]
                if hasattr(est, 'transformer_list'): [inject(s[1]) for s in est.transformer_list]
                if hasattr(est, 'estimator'): inject(est.estimator)
            
            inject(model)
            print(">> SYSTEM: Production Model Loaded & Injected.")
            return
        except Exception as e:
            print(f">> ERROR: Production load failed: {e}")

    # 4. Fallback to Mobile Model
    if os.path.exists('mobile_model.pkl'):
        try:
            model = joblib.load('mobile_model.pkl')
            print(">> SYSTEM: Mobile Model Loaded.")
        except:
            print(">> SYSTEM: Failed to load mobile model.")

# Initialize Logic
load_ai_engine()

# ==========================================
# 4. DATABASE & AUTH SETUP
# ==========================================
def init_db():
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL
            )
        ''')
        conn.commit()

if not os.path.exists(DB_NAME): init_db()

@app.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    return response

# ==========================================
# 5. ROUTES
# ==========================================

@app.route('/predict', methods=['POST'])
def predict():
    if 'user' not in session: return jsonify({'error': 'Unauthorized'}), 401
    
    data = request.get_json()
    text = data.get('text', '')

    if not text: return jsonify({'error': 'Empty data'}), 400

    # --- A. GUARDS ---
    if SPACY_AVAILABLE and nlp_engine:
        doc = nlp_engine(text)
        valid_tokens = [t for t in doc if t.is_alpha and t.has_vector]
        total_tokens = [t for t in doc if t.is_alpha]
        valid_ratio = len(valid_tokens) / len(total_tokens) if total_tokens else 0

        if valid_ratio < 0.4:
            return jsonify({
                'is_fake': True, 'confidence': 99.00, 'fraud_probability': 99.99,
                'reasons': ["🚫 **Gibberish Detected:** Text contains mostly non-English or random words."]
            })

    if any(t in text.upper() for t in ["SELECT *", "DROP TABLE", "<SCRIPT>"]):
        return jsonify({
            'is_fake': True, 'confidence': 99.00, 'fraud_probability': 99.99,
            'reasons': ["🚨 **Security Threat:** Malicious Code / SQL Injection detected."]
        })

    # --- B. PREDICTION ---
    if model:
        try:
            # Predict
            proba = model.predict_proba([text])[0]
            fake_prob = proba[1]
            confidence = max(proba)
            label = "FAKE" if fake_prob > 0.5 else "REAL"

            # LIME XAI
            lime_features = []
            if LIME_AVAILABLE:
                try:
                    explainer = LimeTextExplainer(class_names=['REAL', 'FAKE'])
                    exp = explainer.explain_instance(text, model.predict_proba, num_features=5, num_samples=20) 
                    lime_features = exp.as_list()
                except: pass

            # Human Explanation
            reasons = explain_like_a_human(text, label, lime_features)

            # --- ROUNDING TO 2 DECIMAL PLACES ---
            return jsonify({
                'is_fake': bool(label == "FAKE"),
                'confidence': round(confidence * 100, 2),
                'fraud_probability': round(fake_prob * 100, 2),
                'reasons': reasons
            })

        except Exception as e:
            return jsonify({'error': f"Analysis Error: {str(e)}"}), 500
    else:
        # MOCK MODE
        is_fake = any(w in text.lower() for w in ['urgent', 'wire', 'transfer'])
        reasons = ["⚠️ **Mock Mode:** AI Brain not loaded."]
        if is_fake: reasons.append("🚨 **Keyword Match:** Detected suspicious keywords.")
        
        conf = random.uniform(80, 99)
        return jsonify({
            'is_fake': is_fake, 
            'confidence': round(conf, 2), 
            'fraud_probability': 95.00 if is_fake else 5.00,
            'reasons': reasons,
            'mode': 'mock'
        })

# --- AUTH ENDPOINTS ---
@app.route('/api/signup', methods=['POST'])
def api_signup():
    data = request.get_json()
    hashed_pw = generate_password_hash(data.get('password'), method='pbkdf2:sha256')
    try:
        with sqlite3.connect(DB_NAME) as conn:
            conn.cursor().execute("INSERT INTO users (username, email, password) VALUES (?, ?, ?)", 
                                (data.get('username'), data.get('email'), hashed_pw))
            conn.commit()
        session['user'] = data.get('username')
        return jsonify({'success': True})
    except: return jsonify({'error': 'User exists'}), 409

@app.route('/api/login', methods=['POST'])
def api_login():
    data = request.get_json()
    remember = data.get('remember', False)
    with sqlite3.connect(DB_NAME) as conn:
        row = conn.cursor().execute("SELECT password FROM users WHERE username = ?", (data.get('username'),)).fetchone()
    if row and check_password_hash(row[0], data.get('password')):
        session['user'] = data.get('username')
        session.permanent = remember
        return jsonify({'success': True, 'username': session['user']})
    return jsonify({'error': 'Invalid credentials'}), 401

@app.route('/api/logout', methods=['POST'])
def api_logout(): session.clear(); return jsonify({'success': True})

@app.route('/api/delete_account', methods=['POST'])
def api_delete_account():
    with sqlite3.connect(DB_NAME) as conn:
        conn.cursor().execute("DELETE FROM users WHERE username = ?", (session.get('user'),))
        conn.commit()
    session.clear()
    return jsonify({'success': True})

# --- PAGE ROUTES ---
@app.route('/')
def login_page():
    return redirect(url_for('dashboard_page')) if 'user' in session else render_template('login.html')

@app.route('/dashboard')
def dashboard_page():
    if 'user' not in session: return redirect(url_for('login_page'))
    resp = make_response(render_template('index.html', username=session['user']))
    resp.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    return resp

if __name__ == '__main__':
    app.run(debug=True)
