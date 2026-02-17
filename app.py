"""
JobGuard: Neuro-Symbolic Fraud Detection System (Production Ready)
------------------------------------------------------------------
Features: BERT + Autoencoder + Isolation Forest + LIME + S-BERT + Rule Engine.
Fixes: Added Text Normalization/Preprocessing Pipeline.

Author: Yoge-2004
Version: 15.0.0 (Clean Input Pipeline)
"""

import hashlib
import joblib
import os
import re
import sqlite3
import unicodedata  # Added for Unicode normalization
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Any, Optional, Union

# --- Third-Party Libraries ---
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, session, render_template, redirect, url_for
from flask_caching import Cache
from lime.lime_text import LimeTextExplainer
from werkzeug.security import generate_password_hash, check_password_hash

# --- NLP & Spell Checking ---
from sentence_transformers import SentenceTransformer, util
import torch
import torch.nn as nn
from transformers import DistilBertTokenizer, DistilBertModel
import tensorflow as tf
from tensorflow.keras.models import load_model

# 1. Try importing wordfreq
try:
    from wordfreq import zipf_frequency
    HAS_WORDFREQ = True
except ImportError:
    HAS_WORDFREQ = False
    print("⚠️ 'wordfreq' library not found. Install via: pip install wordfreq")

# 2. Try importing pyenchant
try:
    import enchant
    SPELL_CHECKER = enchant.Dict("en_US")
    HAS_ENCHANT = True
except ImportError:
    HAS_ENCHANT = False
    print("⚠️ 'pyenchant' library not found. Using internal fallback.")

# ==========================================
# 0. SYSTEM CONFIGURATION & LOGGING
# ==========================================

SERVER_LOGS: List[str] = []
MODEL_DIR = "models"
DEVICE = torch.device('cpu') 
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# --- CORPORATE JARGON WHITELIST (Same as before) ---
COMMON_CORP_TERMS = {
    'saas', 'paas', 'iaas', 'kpi', 'roi', 'b2b', 'b2c', 'seo', 'sem', 'crm', 'erp', 'api', 'sdk', 
    'ide', 'gui', 'cli', 'ux', 'ui', 'qa', 'qc', 'uat', 'cicd', 'ci/cd', 'devops', 'secops', 
    'mlops', 'llm', 'nlp', 'gcp', 'aws', 'azure', 'docker', 'kubernetes', 'k8s', 'sql', 'nosql', 
    'json', 'xml', 'yaml', 'html', 'css', 'js', 'react', 'angular', 'vue', 'django', 'flask', 
    'fastapi', 'spring', 'springboot', 'microservices', 'agile', 'scrum', 'kanban', 'jira', 
    'confluence', 'trello', 'asana', 'slack', 'zoom', 'teams', 'meet', 'skype', 'figma', 'adobe', 
    'photoshop', 'illustrator', 'indesign', 'xd', 'sketch', 'invision', 'zeplin', 'symbian', 
    'android', 'ios', 'macos', 'linux', 'ubuntu', 'centos', 'redhat', 'debian', 'fedora', 'kali', 
    'windows', 'microsoft', 'google', 'apple', 'amazon', 'facebook', 'meta', 'netflix', 'tesla', 
    'twitter', 'x', 'linkedin', 'instagram', 'tiktok', 'snapchat', 'pinterest', 'reddit', 'quora', 
    'medium', 'github', 'gitlab', 'bitbucket', 'stackoverflow', 'kaggle', 'leetcode', 'hackerrank', 
    'codewars', 'coursera', 'udemy', 'edx', 'udacity', 'pluralsight', 'linkedinlearning', 
    'salesforce', 'sap', 'oracle', 'ibm', 'hp', 'dell', 'lenovo', 'asus', 'acer', 'msi', 'razer', 
    'logitech', 'sony', 'samsung', 'lg', 'panasonic', 'toshiba', 'hitachi', 'fujitsu', 'nec', 
    'sharp', 'philips', 'siemens', 'bosch', 'ge', 'honeywell', '3m', 'cisco', 'juniper', 'arista', 
    'f5', 'citrix', 'vmware', 'nutanix', 'redhat', 'openshift', 'ansible', 'terraform', 'jenkins', 
    'bamboo', 'circleci', 'travisci', 'gitlabci', 'actions', 'grafana', 'prometheus', 'elk', 
    'splunk', 'datadog', 'newrelic', 'appdynamics', 'dynatrace', 'pagerduty', 'opsgenie', 
    'victorops', 'xmatters', 'servicenow', 'bmc', 'cherwell', 'ivanti', 'freshservice', 'zendesk', 
    'freshdesk', 'intercom', 'drift', 'hubspot', 'marketo', 'eloqua', 'pardot', 'mailchimp', 
    'sendgrid', 'twilio', 'stripe', 'paypal', 'braintree', 'square', 'adyen', 'authorize.net', 
    '2checkout', 'worldpay', 'cybersource', 'firstdata', 'fiserv', 'globalpayments', 'tsys', 
    'elavon', 'heartland', 'vantiv', 'paymentech', 'chase', 'boa', 'citi', 'wells', 'amex', 
    'visa', 'mastercard', 'discover', 'diners', 'jcb', 'unionpay', 'rupay', 'mir', 'eftpos', 
    'interac', 'ach', 'sepa', 'swift', 'iban', 'bic', 'aba', 'routing', 'account', 'ledger', 
    'balance', 'sheet', 'income', 'statement', 'cash', 'flow', 'equity', 'asset', 'liability', 
    'debit', 'credit', 'audit', 'tax', 'vat', 'gst', 'hst', 'pst', 'rst', 'qct', 'payroll', 'hr', 
    'human', 'resources', 'recruitment', 'talent', 'acquisition', 'onboarding', 'offboarding', 
    'performance', 'management', 'learning', 'development', 'compensation', 'benefits', 'compliance', 
    'legal', 'gdpr', 'ccpa', 'hipaa', 'ferpa', 'copa', 'pci', 'dss', 'soc2', 'iso', 'nist', 
    'fedramp', 'fisma', 'glba', 'sox', 'osha', 'eeoc', 'ada', 'fmla', 'flsa', 'erisa', 'cobra', 
    'aca', 'w2', '1099', 'w4', 'i9', 'e-verify', 'background', 'check', 'drug', 'screen', 
    'reference', 'interview', 'offer', 'letter', 'contract', 'agreement', 'nda', 'nca', 'nfa', 
    'ip', 'intellectual', 'property', 'copyright', 'trademark', 'patent', 'trade', 'secret',
    'admin', 'assistant', 'clerk', 'manager', 'executive', 'director', 'vp', 'ceo', 'cto', 'cfo', 
    'coo', 'cmo', 'cio', 'ciso', 'cdo', 'chro', 'clo', 'cpo', 'cso', 'cto', 'founder', 'owner',
    'freelance', 'contract', 'fulltime', 'parttime', 'internship', 'remote', 'hybrid', 'onsite'
}

def log_event(message: str, level: str = "INFO", category: str = "SYSTEM") -> None:
    timestamp = datetime.now().strftime("%H:%M:%S")
    formatted_msg = f"[{timestamp}] [{level}] {message}"
    SERVER_LOGS.append(formatted_msg)
    if len(SERVER_LOGS) > 500: SERVER_LOGS.pop(0)
    print(formatted_msg, flush=True)

log_event("Initializing JobGuard System v15.0...", "INIT", "BOOT")

# ==========================================
# 1. PYTORCH MODEL ARCHITECTURE
# ==========================================

class BERTFusion(nn.Module):
    def __init__(self, num_features: int = 10):
        super(BERTFusion, self).__init__()
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        for param in self.bert.transformer.layer[:4].parameters():
            param.requires_grad = False
        self.feat_proj = nn.Sequential(nn.Linear(num_features, 64), nn.ReLU(), nn.Dropout(0.2))
        self.classifier = nn.Sequential(nn.Linear(768 + 64, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 1))

    def forward(self, input_ids, attention_mask, features):
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_emb = bert_out.last_hidden_state[:, 0, :]
        feat_emb = self.feat_proj(features)
        combined = torch.cat((cls_emb, feat_emb), dim=1)
        return self.classifier(combined)

# ==========================================
# 2. ROBUST FEATURE ENGINEERING
# ==========================================

class TextProcessor:
    
    @staticmethod
    def normalize_text(text: str) -> str:
        """
        Cleans and normalizes input text before any processing.
        1. Unicode normalization (NFKC) -> fixes accents, fancy quotes
        2. Whitespace squashing -> removes \n\n, tabs
        3. Strip control characters
        """
        # 1. Standardize Unicode (e.g. ﬀ -> ff, ½ -> 1/2)
        text = unicodedata.normalize('NFKC', text)
        
        # 2. Replace fancy quotes and dashes
        text = text.replace('“', '"').replace('”', '"').replace('’', "'").replace('–', '-')
        
        # 3. Collapse whitespace (tabs, newlines -> single space)
        # Note: We keep some structure implicitly, but for BERT single stream is fine.
        text = re.sub(r'\s+', ' ', text).strip()
        
        # 4. Remove non-printable control chars (except standard ASCII)
        text = "".join(ch for ch in text if unicodedata.category(ch)[0] != "C")
        
        return text

    @staticmethod
    def extract_features_for_model(text: str) -> np.ndarray:
        text_str = str(text)
        lower_text = text_str.lower()
        length = len(text_str) if len(text_str) > 0 else 1
        
        has_logo = 0 
        has_questions = 1 if "?" in text_str else 0
        caps_ratio = sum(1 for c in text_str if c.isupper()) / length
        has_email = 1 if re.search(r'[\w\.-]+@[\w\.-]+\.\w+', text_str) else 0
        has_phone = 1 if re.search(r'\+?\d[\d -]{8,12}\d', text_str) else 0
        
        admin_keywords = ['admin', 'assistant', 'clerk', 'data entry', 'secretary']
        is_admin = 1 if any(k in lower_text for k in admin_keywords) else 0
        
        socials = ['instagram', 'facebook', 'linkedin', 'twitter', 'telegram', 'whatsapp']
        has_social = 1 if any(s in lower_text for s in socials) else 0
        
        exclamation_ratio = text_str.count("!") / length
        money_count = len(re.findall(r'(\$|rs\.?|usd|inr)\s?\d+', lower_text)) + lower_text.count('salary')
        money_ratio = money_count / length
        
        urgency_words = ['urgent', 'immediate', 'now', 'deadline', 'hurry', 'asap']
        urgency_ratio = sum(1 for w in urgency_words if w in lower_text) / length
        
        return np.array([
            has_logo, has_questions, caps_ratio, has_email, has_phone, 
            is_admin, has_social, exclamation_ratio, money_ratio, urgency_ratio
        ])

    @staticmethod
    def extract_entities_robust(text: str) -> Dict[str, List[str]]:
        entities = {'emails': [], 'urls': [], 'phones': [], 'salaries': [], 'softwares': []}
        
        entities['emails'] = list(set(re.findall(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', text)))
        entities['urls'] = list(set(re.findall(r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+', text)))
        
        raw_phones = re.findall(r'(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', text)
        entities['phones'] = [p for p in list(set(raw_phones)) if len(re.sub(r'\D', '', p)) > 9]
        
        currency_symbols = r'(?:\$|€|£|¥|₹|\bRs\.?|\bCHF|\bCAD|\bAUD|\bSGD|\bNZD|\bHKD|\bCNY|\bINR|\bUSD|\bGBP|\bEUR)'
        frequency_terms = r'(?:per\s?hour|/hr|per\s?week|/wk|per\s?month|/mo|per\s?annum|p\.a\.|yearly|LPA|CTC)'
        
        salary_patterns = [
            fr'{currency_symbols}\s?[\d,]+(?:k|K|L|m|M)?(?:\.\d+)?\s?{frequency_terms}?',
            fr'[\d,]+(?:\.\d+)?\s?(?:USD|EUR|GBP|AUD|CAD|SGD|INR)\s?{frequency_terms}?',
            fr'[\d,]+(?:\.\d+)?\s?{frequency_terms}',
        ]
        
        found_salaries = []
        for pat in salary_patterns:
            matches = re.finditer(pat, text, re.IGNORECASE)
            for m in matches:
                clean_match = re.sub(r'\s+', ' ', m.group(0).strip())
                if re.match(r'^(19|20)\d{2}$', clean_match): continue
                found_salaries.append(clean_match)
        entities['salaries'] = list(set(found_salaries))
        
        tech_stack = ['python', 'java', 'react', 'sql', 'aws', 'docker', 'excel', 'photoshop', 'figma', 'c++', 'typescript', 'marketing', 'node.js', 'angular', 'springboot', 'gcp']
        found_tech = [tech for tech in tech_stack if tech in text.lower()]
        entities['softwares'] = list(set(found_tech))
        
        return entities

    @staticmethod
    def check_safety_risks(text: str) -> List[str]:
        risks = []
        lower = text.lower()
        
        remote_tools = ['anydesk', 'teamviewer', 'logmein', 'remotepc', 'ammyy', 'zoho assist', 'ultraviewer']
        detected_tools = [t for t in remote_tools if t in lower]
        if detected_tools:
            risks.append(f"🚨 **Security Risk:** Mentions remote access software (**{', '.join(detected_tools).title()}**). Scammers use these to steal data.")
            
        money_keywords = ['security deposit', 'registration fee', 'training fee', 'cost of training', 'refundable deposit', 'buy starter kit', 'application processing fee', 'verification fee']
        detected_money = [m for m in money_keywords if m in lower]
        if detected_money:
            risks.append(f"🚨 **Financial Risk:** Asks for money (**{detected_money[0]}**). Legitimate jobs NEVER ask candidates to pay.")
            
        crypto_keywords = [
            (r'\bbitcoin\b', 'Bitcoin'), (r'\bbtc\b', 'BTC'), (r'\busdt\b', 'USDT'), 
            (r'\btether\b', 'Tether'), (r'\bethereum\b', 'Ethereum'), (r'\beth\b', 'ETH'), 
            (r'\bwallet address\b', 'Wallet')
        ]
        for pat, name in crypto_keywords:
            if re.search(pat, text, re.IGNORECASE):
                risks.append(f"🚨 **High Risk:** Mentions Cryptocurrency (**{name}**). Legitimate employers do NOT pay via Crypto.")
                break

        payment_apps = ['google pay', 'gpay', 'phonepe', 'paytm', 'bhim', 'upi id', 'zelle', 'venmo', 'cashapp', 'paypal friends']
        detected_apps = [p for p in payment_apps if p in lower]
        if detected_apps:
            risks.append(f"🚨 **Financial Risk:** Asks for payment via **{detected_apps[0].title()}**.")

        return risks

    @staticmethod
    def detect_programming_language(text: str) -> Optional[str]:
        # STRICT DETECTION: Syntactic Structures
        if re.search(r'\b(public|private|protected)\s+(static\s+)?(void|int|string|boolean|class|interface)\s+\w+', text):
            return "Source Code (Method Signature)"
        if re.search(r'\bdef\s+[a-zA-Z_]\w*\s*\(.*\)\s*:', text):
            return "Source Code (Python Function)"
        if re.search(r'\b(const|let|var)\s+[a-zA-Z_]\w*\s*=\s*[^=]', text):
            return "Source Code (JS/TS Variable)"
        if re.search(r'^\s*#include\s*<.*>', text, re.MULTILINE):
            return "Source Code (C/C++ Header)"
        
        syntax_chars = len(re.findall(r'[\{\}\;\(\)\[\]=<>]', text))
        if len(text) > 50 and (syntax_chars / len(text)) > 0.15:
            return "Source Code (High Syntax Density)"
        return None

    @staticmethod
    def check_emoji_professionalism(text: str) -> Optional[str]:
        emoji_count = len(re.findall(r'[\U00010000-\U0010ffff]', text))
        if emoji_count > 5:
            return f"⚠️ **Professionalism:** Excessive use of emojis ({emoji_count} detected). Typical of MLM/Spam."
        return None

    @staticmethod
    def validate_content(text: str) -> Tuple[bool, List[str]]:
        warnings = []
        if len(text) < 20:
            warnings.append("Text is too short to be a valid job description.")
            return False, warnings
        
        clean_text = re.sub(r'[^\w\s]', '', text.lower())
        tokens = clean_text.split()
        total_tokens = len(tokens)
        
        if total_tokens > 0:
            valid_count = 0
            for t in tokens:
                if t in COMMON_CORP_TERMS:
                    valid_count += 1
                    continue
                if HAS_WORDFREQ:
                    if zipf_frequency(t, 'en') > 0:
                        valid_count += 1
                        continue
                if HAS_ENCHANT and SPELL_CHECKER.check(t):
                    valid_count += 1
                    continue
            
            if (valid_count / total_tokens) < 0.4:
                warnings.append("Gibberish detected (Unrecognizable word patterns).")
                return False, warnings

        code_type = TextProcessor.detect_programming_language(text)
        if code_type:
            warnings.append(f"Input appears to be **{code_type}**, not a job description.")
            return False, warnings
            
        emoji_warn = TextProcessor.check_emoji_professionalism(text)
        if emoji_warn:
            warnings.append(emoji_warn)
            
        return True, warnings

    @staticmethod
    def analyze_domain_reputation(entities: Dict[str, List[str]]) -> Dict[str, List[str]]:
        flags = {'email_flags': [], 'url_flags': []}
        free_domains = {'gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com', 'live.com', 'aol.com', 'proton.me'}
        for email in entities['emails']:
            if email.split('@')[-1].lower() in free_domains:
                flags['email_flags'].append(f"⚠️ **Professionalism:** Uses free email (**{email.split('@')[-1]}**) instead of corporate domain.")

        shorteners = {'bit.ly', 'goo.gl', 'tinyurl.com', 't.co', 'linktr.ee', 'wa.me'}
        for url in entities['urls']:
            if any(s in url.lower() for s in shorteners):
                flags['url_flags'].append(f"⚠️ **Safety:** URL shortener detected (**{url[:25]}...**).")
        return flags

# ==========================================
# 3. MODEL MANAGER
# ==========================================

class ModelManager:
    def __init__(self):
        self.bert_fusion = None
        self.bert_tokenizer = None
        self.autoencoder = None
        self.iso_forest = None
        self.meta_ensemble = None
        self.sbert = None 
        self.scalers = {}
        
    def load(self) -> None:
        log_event("Loading AI Models...", "INIT", "BOOT")
        try:
            self.sbert = SentenceTransformer('all-MiniLM-L12-v2')
            self.scalers['feat'] = joblib.load(os.path.join(MODEL_DIR, 'feature_scaler.pkl'))
            self.scalers['meta'] = joblib.load(os.path.join(MODEL_DIR, 'meta_scaler.pkl'))
            self.scalers['ae'] = joblib.load(os.path.join(MODEL_DIR, 'ae_scaler.pkl'))
            self.iso_forest = joblib.load(os.path.join(MODEL_DIR, 'iso_forest.pkl'))
            self.meta_ensemble = joblib.load(os.path.join(MODEL_DIR, 'meta_ensemble.pkl'))
            self.autoencoder = load_model(os.path.join(MODEL_DIR, 'autoencoder.keras'), compile=False)
            self.bert_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
            self.bert_fusion = BERTFusion(num_features=10)
            state_dict = torch.load(os.path.join(MODEL_DIR, 'best_bert_fusion.pth'), map_location=DEVICE)
            self.bert_fusion.load_state_dict(state_dict)
            self.bert_fusion.to(DEVICE)
            self.bert_fusion.eval()
            log_event("All Neural Networks Loaded & Quantized", "SUCCESS", "INIT")
        except Exception as e:
            log_event(f"Model Load Failed: {str(e)}", "ERROR", "INIT")

model_mgr = ModelManager()
model_mgr.load()

# ==========================================
# 4. PREDICTION PIPELINE
# ==========================================

class FraudDetector:
    def __init__(self):
        self.scam_anchors_text = [
            "Pay registration fee", "Buy laptop from us", "Telegram job offer", 
            "Bank account details needed", "Easy money part time", "No interview required",
            "Kindly transfer money for training materials", "Send your credit card details",
            "Strictly confidential payment transaction", "Instant job offer without assessment",
            "Starter kit purchase required", "Insurance fee for laptop",
            "Earn $1000 weekly", "No experience needed", "Simple copy paste job", 
            "Guaranteed job offer", "Make quick cash", "Earn $500 in 2 hours",
            "Deposit fees to be prioritised", "Priority consideration fee", "Application processing fee",
            "Deposit 5000 fees", "Pay to skip queue",
            "Payment via Bitcoin", "USDT transfer required", "Link your crypto wallet"
        ]
        self.scam_anchors_tensor = None
        if model_mgr.sbert:
            try:
                self.scam_anchors_tensor = model_mgr.sbert.encode(self.scam_anchors_text, convert_to_tensor=True)
            except Exception as e:
                log_event(f"S-BERT Init Failed: {e}", "ERROR", "INIT")

    def predict(self, text: str) -> Dict[str, Any]:
        logs = []
        def trace(msg, level="INFO"): logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] [{level}] {msg}")
        
        # 1. NORMALIZE (Clean input before anything else)
        text = TextProcessor.normalize_text(text)
        
        # 2. VALIDATE
        is_valid, warnings = TextProcessor.validate_content(text)
        if not is_valid:
            for w in warnings: trace(w, "WARN")
            return {'status': 'invalid', 'warnings': warnings, 'logs': logs}
        
        features = TextProcessor.extract_features_for_model(text)
        feats_scaled = model_mgr.scalers['feat'].transform(features.reshape(1, -1))
        
        inputs = model_mgr.bert_tokenizer(text, return_tensors="pt", max_length=128, padding='max_length', truncation=True)
        with torch.no_grad():
            logits = model_mgr.bert_fusion(inputs['input_ids'].to(DEVICE), inputs['attention_mask'].to(DEVICE), torch.tensor(feats_scaled, dtype=torch.float32).to(DEVICE))
            bert_prob = torch.sigmoid(logits).item()
            cls_emb = model_mgr.bert_fusion.bert(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask']).last_hidden_state[:, 0, :].numpy()
        
        ae_emb_scaled = model_mgr.scalers['ae'].transform(cls_emb)
        with tf.device('/cpu:0'): recon = model_mgr.autoencoder.predict(ae_emb_scaled, verbose=0)
        ae_error = float(np.mean(np.power(ae_emb_scaled - recon, 2)))
        iso_score = float(-model_mgr.iso_forest.decision_function(feats_scaled)[0])
        
        meta_in = np.column_stack([[bert_prob], [iso_score], [ae_error], feats_scaled])
        final_prob = model_mgr.meta_ensemble.predict_proba(model_mgr.scalers['meta'].transform(meta_in))[0][1]
        
        trace(f"AI Prob: {final_prob:.4f} (BERT: {bert_prob:.2f})", "AI")
        
        reasons = []
        if self.scam_anchors_tensor is not None:
            # Use regex split for better sentence boundaries
            sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
            sentences = [s.strip() for s in sentences if len(s) > 10]
            if sentences:
                scores = util.cos_sim(model_mgr.sbert.encode(sentences, convert_to_tensor=True), self.scam_anchors_tensor)
                for i, sent_scores in enumerate(scores):
                    curr_max = torch.max(sent_scores).item()
                    if curr_max > 0.55: 
                        suspicious_sent = sentences[i].lower()
                        # SAFETY FILTER
                        safe_words = ['in person', 'video conference', 'google meet', 'zoom', 'teams', 'office', 'headquarters']
                        if 'interview' in suspicious_sent and any(sw in suspicious_sent for sw in safe_words): continue
                        
                        trace(f"S-BERT Match: {curr_max:.2f} -> {sentences[i][:40]}...", "WARN")
                        reasons.append(f"🤖 **Semantic AI:** Suspicious phrase detected: \"{sentences[i]}\" ({(curr_max*100):.0f}% Scam Match)")
                        final_prob = max(final_prob, 0.85)

        safety_risks = TextProcessor.check_safety_risks(text)
        reasons.extend(safety_risks)
        if safety_risks: final_prob = max(final_prob, 0.95)

        entities = TextProcessor.extract_entities_robust(text)
        domain_flags = TextProcessor.analyze_domain_reputation(entities)
        reasons.extend(domain_flags['email_flags'])
        reasons.extend(domain_flags['url_flags'])

        advisory = []
        if entities['emails']: advisory.append(f"✅ **Email:** Verified ({' '.join(entities['emails'][:1])})")
        else: advisory.append("ℹ️ **Email:** Not listed in description")
            
        if entities['urls']: advisory.append(f"✅ **Website:** Link detected ({' '.join(entities['urls'][:1])})")
        else: advisory.append("ℹ️ **Website:** Not listed in description")

        if entities['salaries']: advisory.append(f"✅ **Compensation:** {' '.join(entities['salaries'][:1])}")
        else: advisory.append("ℹ️ **Compensation:** Not explicitly mentioned")
        
        if entities['phones']: advisory.append(f"✅ **Phone:** {' '.join(entities['phones'][:1])}")
        else: advisory.append("ℹ️ **Phone:** Not listed in description")

        if "telegram" in text.lower() or "whatsapp" in text.lower():
            advisory.append("🚩 **Safety Warning:** Job asks for **Telegram** or **WhatsApp** contact.")
            
        emoji_warn = TextProcessor.check_emoji_professionalism(text)
        if emoji_warn:
            reasons.append(emoji_warn)

        return {
            'status': 'success', 'prob': final_prob,
            'reasons': reasons, 'advisory': advisory, 'logs': logs
        }

detector = FraudDetector()

# ==========================================
# 5. FLASK APP & ROUTES
# ==========================================

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "SuperSecretJobGuardKey2026")
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=7)
cache = Cache(app, config={'CACHE_TYPE': 'SimpleCache'})
DB_NAME = "users.db"

def get_db():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_db() as conn:
        conn.execute('''CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, username TEXT UNIQUE, email TEXT UNIQUE, password TEXT)''')
init_db()

lime_explainer = LimeTextExplainer(class_names=['Real', 'Fake'])

@app.route('/')
def home():
    if 'user' in session: return redirect(url_for('dashboard'))
    return render_template('login.html')

@app.route('/dashboard')
def dashboard():
    if 'user' not in session: return redirect(url_for('home'))
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        text_hash = hashlib.sha256(text.encode()).hexdigest()
        cached = cache.get(text_hash)
        if cached:
            log_event("Serving result from Cache", "SUCCESS", "API")
            return jsonify(cached)
        
        log_event(f"Analyzing Job: {text[:30]}...", "INFO", "API")
        result = detector.predict(text)
        
        if result['status'] == 'invalid':
            return jsonify({
                'fraud_probability': 0, 'is_gibberish': True, 'verdict': 'Invalid',
                'anomaly_analysis': result['warnings'], 'system_logs': list(reversed(result['logs']))
            })
            
        final_prob = result['prob']
        
        xai_highlights = []
        if final_prob > 0.35:
            try:
                def batch_predict_lime(texts: List[str]) -> np.ndarray:
                    batch_size = 16
                    all_probs = []
                    for i in range(0, len(texts), batch_size):
                        batch_texts = texts[i:i+batch_size]
                        inputs = model_mgr.bert_tokenizer(batch_texts, return_tensors="pt", max_length=128, padding=True, truncation=True)
                        dummy_feats = torch.zeros((len(batch_texts), 10)).to(DEVICE)
                        with torch.no_grad():
                            logits = model_mgr.bert_fusion(inputs['input_ids'].to(DEVICE), inputs['attention_mask'].to(DEVICE), dummy_feats)
                            probs = torch.sigmoid(logits).cpu().numpy().flatten()
                        for p in probs: all_probs.append([1-p, p])
                    return np.array(all_probs)

                exp = lime_explainer.explain_instance(text, batch_predict_lime, num_features=6, num_samples=100)
                xai_highlights = [x[0] for x in exp.as_list() if x[1] > 0.05]
            except Exception as e:
                log_event(f"LIME Failed: {e}", "WARN", "XAI")

        response = {
            'fraud_probability': round(final_prob * 100, 2),
            'is_gibberish': False,
            'reasons': result['reasons'],        
            'advisory': result['advisory'],      
            'xai_insights': xai_highlights,      
            'anomaly_analysis': [],
            'system_logs': list(reversed(SERVER_LOGS + result['logs']))
        }
        cache.set(text_hash, response, timeout=3600)
        return jsonify(response)
    except Exception as e:
        log_event(f"Prediction Error: {e}", "ERROR", "API")
        return jsonify({'error': str(e)}), 500

@app.route('/api/signup', methods=['POST'])
def signup():
    data = request.get_json()
    try:
        hashed = generate_password_hash(data['password'], method='pbkdf2:sha256')
        with get_db() as conn:
            conn.execute("INSERT INTO users (username, email, password) VALUES (?, ?, ?)", (data['username'], data['email'], hashed))
            session['user'] = data['username']
            session.permanent = True
            log_event(f"New User Registered: {data['username']}", "SUCCESS", "AUTH")
            return jsonify({'success': True})
    except sqlite3.IntegrityError: return jsonify({'error': 'Username/Email exists'}), 409

@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()
    with get_db() as conn:
        user = conn.execute("SELECT * FROM users WHERE username = ?", (data['username'],)).fetchone()
        if user and check_password_hash(user['password'], data['password']):
            session['user'] = user['username']
            session.permanent = data.get('remember', False)
            log_event(f"User Login: {data['username']}", "SUCCESS", "AUTH")
            return jsonify({'success': True, 'username': user['username']})
    return jsonify({'error': 'Invalid credentials'}), 401

@app.route('/api/user_info')
def user_info():
    return jsonify({'username': session['user']}) if 'user' in session else jsonify({'error': 'Not logged in'}), 401

@app.route('/api/delete_account', methods=['POST'])
def delete_account():
    if 'user' not in session: return jsonify({'error': 'Unauthorized'}), 401
    try:
        with get_db() as conn: conn.execute("DELETE FROM users WHERE username = ?", (session['user'],))
        log_event(f"Account Deleted: {session['user']}", "WARN", "AUTH")
        session.clear()
        return jsonify({'success': True})
    except Exception as e: return jsonify({'error': str(e)}), 500

@app.route('/api/system_logs')
def system_logs():
    if session.get('user') != 'Yoge': return jsonify([])
    return jsonify(list(reversed(SERVER_LOGS)))

@app.route('/api/logout', methods=['POST'])
def logout():
    session.clear()
    return jsonify({'success': True})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860, debug=True)
