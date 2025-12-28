# Fake Job Posting and Online Recruitment Scam Detection Using Tri-Core Hybrid AI

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE) [![Python](https://img.shields.io/badge/Python-3.12.12-blue)](https://www.python.org/) [![Framework](https://img.shields.io/badge/Backend-Flask-red)](https://flask.palletsprojects.com/) [![Transformers](https://img.shields.io/badge/🤗_Hugging_Face-DistilBERT-yellow)]() [![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange)]()

## 📋 Table of Contents
1. [📌 Overview](#overview)
2. [✨ Key Features (Application)](#key-features)
3. [🛠️ Tech Stack & Compatibility](#tech-stack)
4. [🔬 Deep Dive: The Tri-Core Architecture](#model-architecture)
5. [📂 Project Structure](#project-structure)
6. [⚙️ Installation & Setup](#installation)
7. [👤 Author](#author)
8. [📜 License](#license)

---

<a id='overview'></a>
## 📌 Overview
This project is a sophisticated **Web-based AI Deception Detector** designed to identify fake job postings and recruitment scams. Unlike static spam filters, this system employs a **Tri-Core Hybrid Intelligence Engine**.

It combines three distinct 'Brains' to analyze fraud from every angle:
1. **Context Brain (DistilBERT):** Understands semantic meaning and vague promises.
2. **Pattern Brain (Ensemble):** Detects high-frequency scam keywords.
3. **Anomaly Brain (Isolation Forest):** Detects statistical outliers and structural irregularities.

The application offers a secure, user-friendly dashboard where candidates can paste job descriptions and receive an instant **Trust Score**, **Risk Breakdown**, and **Explainable AI Insights**.

---

<a id='key-features'></a>
## ✨ Key Features (Application)

### 🛡️ Core Security & Detection
- **Tri-Core AI Engine:** Runs DistilBERT (Context), Sklearn Pipeline (Pattern), and Isolation Forest (Structure) simultaneously.
- **Genius Override Logic:** If DistilBERT is >90% confident, it intelligently overrides weaker models to prevent false alarms.
- **Zero-Day Scam Protection:** The Unsupervised model detects never-before-seen scams by flagging structural irregularities (gibberish, symbol abuse).
- **Behavioral Safety Net:** Catches generic phishing attacks (e.g., *"Click link to verify bank account"*) that AI might miss due to text truncation.

### ⚡ High-Performance Architecture
- **Smart RAM Caching:** Implements `Flask-Caching` to store analysis results. Repeated queries return results in **0.001ms**.
- **Direct Path Loading:** BERT models are loaded from the local root directory for maximum speed and offline capability.

### 🔍 Explainable AI (XAI)
- **LIME Integration:** Explains *which words* triggered the BERT fraud score.
- **Anomaly Explanation:** Explains *why* the structure is bad (e.g., *"Statistical Structural Outlier detected"*).

---

<a id='tech-stack'></a>
## 🛠️ Tech Stack & Compatibility

**Core Environment:**
- **Python:** 3.12.12 (Strict Requirement)
- **PyTorch:** 2.x (For BERT Inference)
- **TensorFlow:** 2.19 (For Autoencoders)
- **Scikit-Learn:** 1.6.1

**AI Engines:**
- **Transformer:** `distilbert-base-uncased` (Hugging Face)
- **NLP:** Spacy (`en_core_web_lg`)
- **Anomaly Detection:** Isolation Forest + Deep Autoencoders
- **Interpretation:** LIME (Local Interpretable Model-agnostic Explanations)

**Backend & Frontend:**
- **Framework:** Flask (Python)
- **UI:** HTML5, CSS3 (Glassmorphism), JavaScript (Fetch API)

---

<a id='model-architecture'></a>
## 🔬 Deep Dive: The Tri-Core Architecture
This system uses a novel **Ensemble Voting Approach**, running three independent AI brains to ensure 360° protection.

### 🧠 Brain 1: The Context Engine (DistilBERT)
*Objective: To understand the meaning and intent of the text.*
- **Model:** Fine-tuned DistilBERT Classifier (PyTorch).
- **Input:** Tokenized Text (Max 512 Tokens).
- **Mechanism:** Detects subtle semantic cues (e.g., vague promises, inconsistent logic) that keyword counters miss.
- **Weight:** Contributes **60%** to the final decision (or 100% if confidence > 90%).

### 📊 Brain 2: The Pattern Engine (Supervised Ensemble)
*Objective: To count 'Red Flag' keywords.*
- **Model:** Voting Classifier (MLP + Gradient Boosting + Logistic Regression).
- **Input:** TF-IDF Vectors + Spacy Word Vectors.
- **Mechanism:** Detects high-frequency scam words (e.g., "Urgent", "Wire Transfer", "WhatsApp").
- **Weight:** Contributes **25%** to the final decision.

### 🦄 Brain 3: The Anomaly Engine (Isolation Forest & Autoencoder)
*Objective: To analyze HOW it is written (Syntax & Structure).* 
- **Models:** **Isolation Forest** (Primary) + Deep Autoencoder (Secondary).
- **Input:** 307 Features (300 Semantic + 7 Structural Ratios).
- **Mechanism (Isolation Forest):** Randomly selects a feature and splits values. Anomalies (scams) are isolated quickly because they are rare and different from the 'Real Job' distribution.
- **Mechanism (Autoencoder):** Reconstructs text features; high error means the text is 'weird' or 'gibberish'.
- **Role:** Acts as a booster. If active, it adds +15% to the Fraud Probability.

---

<a id='project-structure'></a>
## 📂 Project Structure
The project files are organized as follows:

```bash
Fake_Job_Detection_Python/
│
├── app.py                            # Main Flask Application (The Engine)
├── requirements.txt                  # Dependencies (Includes Spacy Model URL)
├── README.md                         # This Documentation
├── LICENSE                           # MIT License
├── test.py                           # Accuracy Validation Script
│
├── config.json                       # BERT Architecture Config
├── model.safetensors                 # BERT Weights (The Brain - ~260MB)
├── tokenizer.json                    # BERT Tokenizer Data
├── tokenizer_config.json             # BERT Tokenizer Settings (CRITICAL)
├── vocab.txt                         # BERT Vocabulary List
├── special_tokens_map.json           # BERT Special Token Rules
│
├── production_fake_job_pipeline.pkl  # Sklearn Supervised Model
├── robust_anomaly_model.pkl          # Isolation Forest & Autoencoder Model
│
├── users.db                          # User Credentials Database (Auto-generated)
├── fake_job_postings.csv             # Raw Dataset for Training
├── results.csv                       # Test Results Log
│
├── templates/                        # Frontend Views
│   ├── index.html                    # Main Dashboard
│   └── login.html                    # Auth Pages
│
└── static/                           # Styles & Scripts
    ├── css/
    │   ├── style.css                 # Glassmorphism Dashboard Styles
    │   └── login.css                 # Login Animation Styles
    └── js/
        ├── script.js                 # Dashboard Logic (API & LIME Rendering)
        └── login.js                  # Auth Logic
```

---

<a id='installation'></a>
## ⚙️ Installation & Setup

**1. Clone the Repository**
```bash
git clone [https://github.com/Yoge-2004/Fake_Job_Detection_Python.git](https://github.com/Yoge-2004/Fake_Job_Detection_Python.git)
cd Fake_Job_Detection_Python
```

**2. Install Dependencies**
This command installs Flask, TensorFlow, PyTorch, Transformers, AND the Spacy English model automatically.
```bash
pip install -r requirements.txt
```

**3. Verify Model Files**
Ensure `model.safetensors` and `config.json` are present in the root directory (same folder as `app.py`).

**4. Run the App**
```bash
python app.py
```
Visit `http://127.0.0.1:5000` in your browser.

---

<a id='author'></a>
## 👤 Author

**Yogeshwaran**
- **Role:** AI & Full Stack Developer
- **GitHub:** [https://github.com/Yoge-2004](https://github.com/Yoge-2004)
- **Project:** JobGuard AI (Final Year Project 2025)

---

<a id='license'></a>
## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.

---
Made with ❤️, Python, and Transformers.
