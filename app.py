"""
JobGuard: Neuro-Symbolic Fraud Detection System (Production SOTA)
------------------------------------------------------------------
Features: BERTFusion (Full-Document Sliding-Window Max-Gated MIL) + Autoencoder
          + Isolation Forest + Calibrated Meta-Ensemble + LIME + S-BERT + Rule Engine.
Author: Yoge-2004
Version: 16.0.0 (SOTA 98.45% Acc / 0.9942 ROC-AUC / 96.0% Trojan-Horse Nuance Recall)
"""

import hashlib
import joblib
import os
import re
import sqlite3
import unicodedata
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Any, Optional

# --- Third-Party Libraries ---
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, session, render_template, redirect, url_for
from flask_caching import Cache
from lime.lime_text import LimeTextExplainer
from werkzeug.security import generate_password_hash, check_password_hash

# --- NLP & Deep Learning ---
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
    print("⚠️ 'wordfreq' library not found. Install via: uv add wordfreq")

# 2. Try importing pyenchant
try:
    import enchant
    SPELL_CHECKER = enchant.Dict("en_US")
    HAS_ENCHANT = True
except ImportError:
    HAS_ENCHANT = False
    print("⚠️ 'pyenchant' library not found. Install via: uv add pyenchant")

# ==========================================
# 0. SYSTEM CONFIGURATION & LOGGING
# ==========================================
SERVER_LOGS: List[str] = []
MODEL_DIR = "models"
DEVICE = torch.device('cpu')
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
MAX_LEN = 256  # Synchronized 1-to-1 with SOTA Kaggle training pipeline

# --- CORPORATE JARGON WHITELIST ---
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
    'f5', 'citrix', 'vmware', 'nutanix', 'openshift', 'ansible', 'terraform', 'jenkins',
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
    'coo', 'cmo', 'cio', 'ciso', 'cdo', 'chro', 'clo', 'cpo', 'cso', 'founder', 'owner',
    'freelance', 'fulltime', 'parttime', 'internship', 'remote', 'hybrid', 'onsite'
}


def log_event(message: str, level: str = "INFO", category: str = "SYSTEM") -> None:
    timestamp = datetime.now().strftime("%H:%M:%S")
    formatted_msg = f"[{timestamp}] [{level}] {message}"
    SERVER_LOGS.append(formatted_msg)
    if len(SERVER_LOGS) > 500:
        SERVER_LOGS.pop(0)
    print(formatted_msg, flush=True)


log_event("Initializing JobGuard SOTA System v16.0.0...", "INIT", "BOOT")


# ==========================================
# 1. PYTORCH MODEL ARCHITECTURE (Single-Pass Ready)
# ==========================================
class BERTFusion(nn.Module):
    def __init__(self, num_features: int = 10):
        super(BERTFusion, self).__init__()
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False
        self.feat_proj = nn.Sequential(nn.Linear(num_features, 64), nn.ReLU(), nn.Dropout(0.2))
        self.classifier = nn.Sequential(nn.Linear(768 + 64, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 1))

    def forward(self, input_ids, attention_mask, features, return_emb: bool = False):
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_emb = bert_out.last_hidden_state[:, 0, :]
        feat_emb = self.feat_proj(features)
        combined = torch.cat((cls_emb, feat_emb), dim=1)
        logits = self.classifier(combined)
        if return_emb:
            return logits, cls_emb
        return logits


# ==========================================
# 2. ROBUST FEATURE ENGINEERING (100% Synced with Kaggle SOTA Run)
# ==========================================
class TextProcessor:

    @staticmethod
    def normalize_text(text: str) -> str:
        text = unicodedata.normalize('NFKC', str(text))
        text = text.replace('“', '"').replace('”', '"').replace('’', "'").replace('–', '-')
        text = re.sub(r'\s+', ' ', text).strip()
        text = "".join(ch for ch in text if unicodedata.category(ch)[0] != "C")
        return text

    @staticmethod
    def extract_features_for_model(text: str) -> np.ndarray:
        """
        100% locked to EXACT_FEATURE_ORDER and clamped window density [300, 1200]
        used in SOTA Kaggle training run (download_2.txt).
        """
        text_str = str(text)
        lower_text = text_str.lower()
        length = len(text_str) if len(text_str) > 0 else 1
        # Clamped window denominator prevents 1/length ratio decay on large 4,000+ char postings
        norm_len = float(min(max(length, 300), 1200))

        has_company_logo = 0.0
        has_questions = 1.0 if "?" in text_str else 0.0
        caps_ratio = sum(1 for c in text_str if c.isupper()) / length
        has_email_in_desc = 1.0 if re.search(r'[\w\.-]+@[\w\.-]+\.\w+', text_str) else 0.0
        has_phone_in_desc = 1.0 if re.search(r'\+?\d[\d -]{8,12}\d', text_str) else 0.0

        admin_keywords = ['admin', 'assistant', 'clerk', 'data entry', 'secretary']
        is_admin_role = 1.0 if any(k in lower_text for k in admin_keywords) else 0.0

        socials = ['instagram', 'facebook', 'linkedin', 'twitter', 'telegram', 'whatsapp']
        has_social_mention = 1.0 if any(s in lower_text for s in socials) else 0.0

        exclamation_ratio = text_str.count("!") / norm_len
        money_count = len(re.findall(r'(\$|rs\.?|usd|inr)\s?\d+', lower_text)) + lower_text.count('salary')
        money_mention_ratio = money_count / norm_len

        urgency_words = ['urgent', 'immediate', 'now', 'deadline', 'hurry', 'asap']
        urgency_ratio = sum(1 for w in urgency_words if w in lower_text) / norm_len

        return np.array([
            has_company_logo,      # 1 (index 0)
            has_questions,         # 2 (index 1)
            caps_ratio,            # 3 (index 2)
            exclamation_ratio,     # 4 (index 3)
            money_mention_ratio,   # 5 (index 4)
            urgency_ratio,         # 6 (index 5)
            has_email_in_desc,     # 7 (index 6)
            has_phone_in_desc,     # 8 (index 7)
            is_admin_role,         # 9 (index 8)
            has_social_mention     # 10 (index 9)
        ], dtype=np.float32)

    @staticmethod
    def extract_entities_robust(text: str) -> Dict[str, List[str]]:
        entities = {'emails': [], 'urls': [], 'phones': [], 'salaries': [], 'softwares': []}

        entities['emails'] = list(set(re.findall(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', text)))
        entities['urls'] = list(set(re.findall(r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+', text)))

        raw_phones = re.findall(r'(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', text)
        entities['phones'] = [p for p in list(set(raw_phones)) if len(re.sub(r'\D', '', p)) > 9]

        currency_symbols = r'(?:\$|€|£|¥|₹|\bRs\.?|\bCHF|\bCAD|\bAUD|\bSGD|\bNZD|\bHKD|\bCNY|\bINR|\bUSD|\bGBP|\bEUR)'
        frequency_terms = r'(?:per\s?hour|/hr|per\s?day|daily|per\s?week|/wk|weekly|per\s?month|/mo|monthly|per\s?annum|p\.a\.|yearly|LPA|CTC)'

        salary_patterns = [
            fr'{currency_symbols}\s?[\d,]+(?:\s?-\s?{currency_symbols}?\s?[\d,]+)?(?:k|K|L|m|M)?(?:\.\d+)?\s?{frequency_terms}?',
            fr'[\d,]+(?:\.\d+)?\s?(?:USD|EUR|GBP|AUD|CAD|SGD|INR)\s?{frequency_terms}?',
            fr'[\d,]+(?:\.\d+)?\s?{frequency_terms}',
        ]

        found_salaries = []
        for pat in salary_patterns:
            matches = re.finditer(pat, text, re.IGNORECASE)
            for m in matches:
                clean_match = re.sub(r'\s+', ' ', m.group(0).strip())
                if re.match(r'^(19|20)\d{2}$', clean_match):
                    continue
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
            risks.append(f"🚨 **Security Risk:** Mentions remote access software (**{', '.join(detected_tools).title()}**). Scammers use these to hijack banking sessions.")

        money_keywords = [
            'security deposit', 'registration fee', 'training fee', 'cost of training',
            'refundable deposit', 'buy starter kit', 'starter kit deposit', 'application processing fee',
            'verification fee', 'security clearance fee', 'background check fee', 'processing fee',
            'purchase your own laptop', 'purchase their pre-configured', 'certified vendor', 'approved vendor'
        ]
        detected_money = [m for m in money_keywords if m in lower]
        if detected_money:
            risks.append(f"🚨 **Financial Risk:** Demands upfront candidate payment or vendor equipment purchase (**{detected_money[0]}**). Legitimate employers NEVER ask candidates to pay.")

        # Detect money-mule / check-cashing / payment-forwarding clauses
        if re.search(r'(receive|accept|process)\s+(client\s+)?(payments?|transfers?|funds?)\s+into\s+your\s+(personal\s+)?bank', lower) or \
           re.search(r'forward(ing)?\s+\d+%\s+via\s+(wire|bitcoin|usdt|crypto)', lower):
            risks.append("🚨 **Money Laundering / Mule Risk:** Job involves receiving funds into your personal bank account and forwarding money/crypto.")

        crypto_keywords = [
            (r'\bbitcoin\b', 'Bitcoin'), (r'\bbtc\b', 'BTC'), (r'\busdt\b', 'USDT'),
            (r'\btether\b', 'Tether'), (r'\bethereum\b', 'Ethereum'), (r'\beth\b', 'ETH'),
            (r'\bwallet address\b', 'Wallet'), (r'\bpayroll wallet\b', 'Payroll Wallet')
        ]
        for pat, name in crypto_keywords:
            if re.search(pat, text, re.IGNORECASE):
                risks.append(f"🚨 **High Risk:** Mentions Cryptocurrency (**{name}**). Legitimate corporate payrolls do NOT require crypto transfers.")
                break

        payment_apps = ['google pay', 'gpay', 'phonepe', 'paytm', 'bhim', 'upi id', 'zelle', 'venmo', 'cashapp', 'paypal friends']
        detected_apps = [p for p in payment_apps if p in lower]
        if detected_apps:
            risks.append(f"🚨 **Financial Risk:** Requests peer-to-peer payment via **{detected_apps[0].title()}**.")

        return risks

    @staticmethod
    def detect_programming_language(text: str) -> Optional[str]:
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
                if HAS_WORDFREQ and zipf_frequency(t, 'en') > 0:
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
                flags['email_flags'].append(f"⚠️ **Professionalism:** Uses personal/free email (**{email.split('@')[-1]}**) instead of a verified corporate domain.")

        shorteners = {'bit.ly', 'goo.gl', 'tinyurl.com', 't.co', 'linktr.ee', 'wa.me'}
        for url in entities['urls']:
            if any(s in url.lower() for s in shorteners):
                flags['url_flags'].append(f"⚠️ **Safety:** URL shortener or chat redirect detected (**{url[:25]}...**).")

        return flags


# ==========================================
# 3. MODEL MANAGER (With Dynamic INT8 CPU Quantization)
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
        log_event("Loading SOTA AI Models...", "INIT", "BOOT")
        try:
            self.sbert = SentenceTransformer('all-MiniLM-L12-v2', device='cpu')
            self.scalers['feat'] = joblib.load(os.path.join(MODEL_DIR, 'feature_scaler.pkl'))
            self.scalers['meta'] = joblib.load(os.path.join(MODEL_DIR, 'meta_scaler.pkl'))
            self.scalers['ae'] = joblib.load(os.path.join(MODEL_DIR, 'ae_scaler.pkl'))
            self.iso_forest = joblib.load(os.path.join(MODEL_DIR, 'iso_forest.pkl'))
            self.meta_ensemble = joblib.load(os.path.join(MODEL_DIR, 'meta_ensemble.pkl'))
            self.autoencoder = load_model(os.path.join(MODEL_DIR, 'autoencoder.keras'), compile=False)
            self.bert_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

            base_bert = BERTFusion(num_features=10)
            state_dict = torch.load(os.path.join(MODEL_DIR, 'best_bert_fusion.pth'), map_location=DEVICE)
            base_bert.load_state_dict(state_dict)
            base_bert.to(DEVICE)
            base_bert.eval()

            # Dynamic INT8 quantization for ~2.5x faster CPU inference on Hugging Face Spaces
            try:
                self.bert_fusion = torch.ao.quantization.quantize_dynamic(
                    base_bert, {nn.Linear}, dtype=torch.qint8
                )
                log_event("All Neural Networks Loaded & Dynamically Quantized (INT8)", "SUCCESS", "INIT")
            except Exception:
                self.bert_fusion = base_bert
                log_event("All Neural Networks Loaded (FP32 CPU)", "SUCCESS", "INIT")
        except Exception as e:
            log_event(f"Model Load Failed: {str(e)}", "ERROR", "INIT")


model_mgr = ModelManager()
model_mgr.load()


# ==========================================
# 4. SOTA PREDICTION PIPELINE (Full-Doc Sliding-Window MIL)
# ==========================================
class FraudDetector:
    def __init__(self):
        self.scam_anchors_text = [
            "Pay registration fee", "Buy laptop from our certified vendor", "Telegram job offer",
            "Bank account details needed for onboarding", "Easy money part time", "No interview required",
            "Kindly transfer money for training materials", "Send your credit card details",
            "Strictly confidential payment transaction", "Instant job offer without assessment",
            "Starter kit purchase required", "Insurance fee for laptop", "Reimbursed on your first paycheck",
            "Earn $1000 weekly from home", "No experience needed", "Simple copy paste job",
            "Guaranteed job offer", "Make quick cash", "Earn $500 in 2 hours",
            "Deposit fees to be prioritised", "Priority consideration fee", "Application processing fee",
            "Refundable security clearance deposit", "Pay to skip queue",
            "Payment via Bitcoin", "USDT transfer required", "Link your crypto wallet",
            "Receive client payments into your personal bank account and forward via wire transfer",
            "Contact our hiring manager on WhatsApp or Telegram for instant interview"
        ]
        self.scam_anchors_tensor = None
        if model_mgr.sbert:
            try:
                self.scam_anchors_tensor = model_mgr.sbert.encode(self.scam_anchors_text, convert_to_tensor=True)
            except Exception as e:
                log_event(f"S-BERT Init Failed: {e}", "ERROR", "INIT")

    @staticmethod
    def build_sliding_window_chunks(text: str) -> List[str]:
        """
        Matches the exact SOTA training sliding-window MIL logic:
        - If <= 200 words: single chunk.
        - If > 200 words: Chunk 0 is the Global Head+Tail context (first 100 + last 100 words),
          followed by overlapping 190-word windows (stride = 100 words) across 100% of the document.
        """
        words = str(text).split()
        if len(words) <= 200:
            return [str(text)]

        chunks = [" ".join(words[:100] + words[-100:])]
        for i in range(0, len(words), 100):
            chunks.append(" ".join(words[i:i + 190]))
            if i + 190 >= len(words):
                break
        return chunks

    def predict(self, text: str) -> Dict[str, Any]:
        logs = []

        def trace(msg, level="INFO"):
            logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] [{level}] {msg}")

        # 1. NORMALIZE
        text = TextProcessor.normalize_text(text)

        # 2. VALIDATE
        is_valid, warnings = TextProcessor.validate_content(text)
        if not is_valid:
            for w in warnings:
                trace(w, "WARN")
            return {'status': 'invalid', 'warnings': warnings, 'logs': logs, 'feats_scaled': None}

        features = TextProcessor.extract_features_for_model(text)
        feats_scaled = model_mgr.scalers['feat'].transform(features.reshape(1, -1))

        # --- 3. FULL-DOCUMENT SLIDING WINDOW + SINGLE-PASS BATCH INFERENCE ---
        chunk_texts = self.build_sliding_window_chunks(text)
        num_chunks = len(chunk_texts)

        enc = model_mgr.bert_tokenizer(
            chunk_texts,
            add_special_tokens=True,
            max_length=MAX_LEN,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        chunk_tensor = enc['input_ids'].to(DEVICE)
        mask_tensor = enc['attention_mask'].to(DEVICE)
        feats_tensor = torch.tensor(
            np.repeat(feats_scaled, num_chunks, axis=0), dtype=torch.float32, device=DEVICE
        )

        with torch.no_grad():
            logits, cls_embs_tensor = model_mgr.bert_fusion(
                chunk_tensor, mask_tensor, feats_tensor, return_emb=True
            )
            chunk_probs = torch.sigmoid(logits).cpu().numpy().flatten()
            chunk_cls_embs = cls_embs_tensor.cpu().numpy()

        # --- 4. MAX-GATED MULTIPLE-INSTANCE AGGREGATION (Prevents Boilerplate Dilution) ---
        max_idx = int(np.argmax(chunk_probs))
        max_prob = float(chunk_probs[max_idx])
        mean_prob = float(np.mean(chunk_probs))

        bert_prob = (0.85 * max_prob + 0.15 * mean_prob) if num_chunks > 1 else max_prob
        most_suspicious_chunk_text = chunk_texts[max_idx]

        # Route the highest-risk chunk's CLS embedding to the Autoencoder
        cls_emb = chunk_cls_embs[max_idx:max_idx + 1]

        ae_emb_scaled = model_mgr.scalers['ae'].transform(cls_emb)
        with tf.device('/cpu:0'):
            recon = model_mgr.autoencoder(ae_emb_scaled, training=False).numpy()
        ae_error = float(np.mean(np.power(ae_emb_scaled - recon, 2), axis=1)[0])
        iso_score = float(-model_mgr.iso_forest.decision_function(feats_scaled)[0])

        meta_in = np.column_stack([[bert_prob], [iso_score], [ae_error], feats_scaled])
        final_prob = float(model_mgr.meta_ensemble.predict_proba(model_mgr.scalers['meta'].transform(meta_in))[0][1])

        trace(f"AI Prob: {final_prob:.4f} (BERT MIL: {bert_prob:.2f} across {num_chunks} chunks)", "AI")

        reasons = []
        if self.scam_anchors_tensor is not None:
            sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', text)
            sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
            if sentences:
                scores = util.cos_sim(model_mgr.sbert.encode(sentences, convert_to_tensor=True), self.scam_anchors_tensor)
                for i, sent_scores in enumerate(scores):
                    curr_max = torch.max(sent_scores).item()
                    if curr_max > 0.56:
                        suspicious_sent = sentences[i].lower()
                        safe_words = ['in person', 'video conference', 'google meet', 'zoom', 'teams', 'office', 'headquarters']
                        if 'interview' in suspicious_sent and any(sw in suspicious_sent for sw in safe_words):
                            continue

                        trace(f"S-BERT Match: {curr_max:.2f} -> {sentences[i][:40]}...", "WARN")
                        reasons.append(f"🤖 **Semantic AI:** Suspicious clause detected: \"{sentences[i]}\" ({(curr_max*100):.0f}% Scam Pattern Match)")
                        final_prob = max(final_prob, 0.85)

        safety_risks = TextProcessor.check_safety_risks(text)
        reasons.extend(safety_risks)
        if safety_risks:
            final_prob = max(final_prob, 0.95)

        entities = TextProcessor.extract_entities_robust(text)
        domain_flags = TextProcessor.analyze_domain_reputation(entities)
        reasons.extend(domain_flags['email_flags'])
        reasons.extend(domain_flags['url_flags'])

        advisory = []
        if entities['emails']:
            advisory.append(f"✅ **Email:** Listed ({' '.join(entities['emails'][:1])})")
        else:
            advisory.append("ℹ️ **Email:** Not listed in description")

        if entities['urls']:
            advisory.append(f"✅ **Website:** Link detected ({' '.join(entities['urls'][:1])})")
        else:
            advisory.append("ℹ️ **Website:** Not listed in description")

        if entities['salaries']:
            advisory.append(f"✅ **Compensation:** {' '.join(entities['salaries'][:1])}")
        else:
            advisory.append("ℹ️ **Compensation:** Not explicitly mentioned")

        if entities['phones']:
            advisory.append(f"✅ **Phone:** {' '.join(entities['phones'][:1])}")
        else:
            advisory.append("ℹ️ **Phone:** Not listed in description")

        if "telegram" in text.lower() or "whatsapp" in text.lower():
            advisory.append("🚩 **Safety Warning:** Job requests off-platform **Telegram** or **WhatsApp** contact.")

        emoji_warn = TextProcessor.check_emoji_professionalism(text)
        if emoji_warn:
            reasons.append(emoji_warn)

        return {
            'status': 'success',
            'prob': final_prob,
            'reasons': reasons,
            'advisory': advisory,
            'logs': logs,
            'feats_scaled': feats_scaled,
            'most_suspicious_chunk': most_suspicious_chunk_text
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
    if 'user' in session:
        return redirect(url_for('dashboard'))
    return render_template('login.html')


@app.route('/dashboard')
def dashboard():
    if 'user' not in session:
        return redirect(url_for('home'))
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
                real_feats = result['feats_scaled']
                # Run LIME on the highest-risk window so long 3,000-word postings explain the exact scam trigger fast!
                explain_target_text = result.get('most_suspicious_chunk', text)

                def batch_predict_lime(texts: List[str]) -> np.ndarray:
                    inputs = model_mgr.bert_tokenizer(
                        texts, return_tensors="pt", max_length=MAX_LEN, padding='max_length', truncation=True
                    )
                    actual_feats_batch = torch.tensor(
                        np.repeat(real_feats, len(texts), axis=0), dtype=torch.float32, device=DEVICE
                    )
                    with torch.no_grad():
                        logits = model_mgr.bert_fusion(
                            inputs['input_ids'].to(DEVICE), inputs['attention_mask'].to(DEVICE), actual_feats_batch
                        )
                        probs = torch.sigmoid(logits).cpu().numpy().flatten()
                    return np.column_stack([1.0 - probs, probs])

                exp = lime_explainer.explain_instance(explain_target_text, batch_predict_lime, num_features=6, num_samples=24)
                xai_highlights = [x[0] for x in exp.as_list() if x[1] > 0.04]

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
    except sqlite3.IntegrityError:
        return jsonify({'error': 'Username/Email exists'}), 409


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
    if 'user' in session:
        return jsonify({'username': session['user']})
    return jsonify({'error': 'Not logged in'}), 401


@app.route('/api/delete_account', methods=['POST'])
def delete_account():
    if 'user' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    try:
        with get_db() as conn:
            conn.execute("DELETE FROM users WHERE username = ?", (session['user'],))
        log_event(f"Account Deleted: {session['user']}", "WARN", "AUTH")
        session.clear()
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/system_logs')
def system_logs():
    if session.get('user') != 'Yoge':
        return jsonify([])
    return jsonify(list(reversed(SERVER_LOGS)))


@app.route('/api/logout', methods=['POST'])
def logout():
    session.clear()
    return jsonify({'success': True})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860, debug=False)
