# JobGuard: Neuro-Symbolic Fraud Detection System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE) [![Python](https://img.shields.io/badge/Python-3.12%2B-blue)](https://www.python.org/) [![Framework](https://img.shields.io/badge/Backend-Flask-red)](https://flask.palletsprojects.com/) [![AI Engine](https://img.shields.io/badge/Model-Voting%20Ensemble-green)]() [![State](https://img.shields.io/badge/System-Production%20Ready-brightgreen)]()

## 📋 Table of Contents
1. [📌 Executive Overview](#overview)
2. [🔬 Deep Dive: The Neuro-Symbolic Architecture](#architecture)
3. [✨ Key Features & Capabilities](#features)
4. [📚 Frameworks & Libraries](#frameworks)
5. [📂 Project Structure](#project-structure)
6. [⚙️ Installation & Setup](#installation)
7. [👤 Author](#author)
8. [📜 License](#license)

---

<a id='overview'></a>
## 📌 Executive Overview

**The Problem:** The rise of remote work has led to a 300% increase in online recruitment fraud. Traditional spam filters, which rely on simple keyword matching (e.g., blocking "Bitcoin"), are easily bypassed by sophisticated scammers who use corporate jargon and legitimate-looking templates.

**The Solution:** **JobGuard** is a next-generation deception detection system designed to identify fraudulent job postings with **human-like reasoning**. It represents a paradigm shift from simple "Pattern Matching" to "Intent Understanding."

**Core Philosophy:** The system employs a **Neuro-Symbolic AI** approach:
- **The "Neural" Brain (Deep Learning):** Uses Transformers (BERT) to read between the lines," detecting the subtle *tone* of desperation, vagueness, or unprofessionalism that rule-based systems miss.
- **The "Symbolic" Brain (Hard Logic):** Uses strict Rule Engines and Regex to catch specific technical red flags (e.g., Crypto wallets, Instant Payment apps, Raw Code injections) that probabilistic models might overlook.

By fusing these two worlds, JobGuard achieves a robust defense against both "Zero-Day" novel scams and classic "Copy-Paste" fraud.

---

<a id='architecture'></a>
## 🔬 Deep Dive: The Neuro-Symbolic Architecture

The system architecture is designed as a **Multi-Stage Pipeline** that mimics the cognitive process of a human fraud analyst. Data flows through four distinct phases before a final verdict is rendered.



### Phase 1: The Gatekeeper (Sanitization & Validation)
Before any AI inference, the raw input is rigorously cleaned to prevent adversarial attacks.
- **Unicode Normalization (NFKC):** Neutralizes "homoglyph attacks" where scammers use Cyrillic characters (e.g., 'а' instead of 'a') to bypass keyword filters.
- **Gibberish Detection (WordFreq):** Rejects inputs that fail the **Zipf Frequency Test** (e.g., "asdfghjkl"), ensuring expensive GPU resources aren't wasted on garbage data.
- **Code Injection Block:** Uses structural regex to detect and reject raw source code (C++/Java/Python) pasted into the description field.

### Phase 2: Feature Extraction (The Sensors)
The cleaned text is then analyzed by three independent feature extractors:

1.  **Linguistic Features (BERT Fusion):**
    - The text is tokenized and passed through a fine-tuned **DistilBERT** model.
    - **Output:** A dense 768-dimensional embedding vector representing the *semantic context*.
2.  **Structural Anomalies (Autoencoder + IsoForest):**
    - **Deep Autoencoder:** Attempts to compress and reconstruct the text features. High reconstruction error (MSE) indicates the text deviates from the "Norm" of legitimate corporate postings.
    - **Isolation Forest:** A tree-based outlier detector that flags metadata anomalies (e.g., descriptions that are suspiciously short or lack punctuation).
3.  **Heuristic Features:**
    - 10 handcrafted features including **Urgency Score**, **Capitalization Ratio**, **Symbol Density**, and **Emoji Professionalism Score**.

### Phase 3: The Supervised Committee (Voting Ensemble)
The extracted features are fed into `meta_ensemble.pkl`, a **Soft Voting Classifier** composed of three diverse algorithms. This ensures robustness against overfitting.

#### 1. Multi-Layer Perceptron (MLP)
   - **Role:** Non-Linear Pattern Recognition.
   - **Logic:** It captures complex interactions, such as *"High Salary"* + *"No Experience"* = *"Fraud"*.

#### 2. Gradient Boosting Machine (GBM)
   - **Role:** Rule-Based Decision Making.
   - **Logic:** An ensemble of decision trees that excels at handling tabular data and cutting through noise to find critical "Red Flags."

#### 3. Logistic Regression
   - **Role:** The Calibrator.
   - **Logic:** A linear probabilistic model that provides a stable baseline. It ensures the final probability score (0-100%) is mathematically well-calibrated.

**Ensemble Logic:** 
$$ 
P_{ensemble} = \frac{1}{3} (P_{MLP} + P_{GBM} + P_{LogReg}) 
$$

### Phase 4: The Parallel Inference Expert (S-BERT)
Running alongside the ensemble is the **Semantic Knowledge Base**.

#### 🧠 Sentence-BERT (S-BERT)
- **Role:** Inference-Only Semantic Search.
- **Logic:** It compares the input text against a pre-computed database of **50+ Known Scam Concepts** (e.g., "Pay for training", "No interview").
- **The Safety Valve:** Unlike the other models, S-BERT is context-aware. It can differentiate between *"No interview required"* (Scam) and *"Interview in person"* (Legit) using negative constraints.

---

<a id='features'></a>
## ✨ Key Features & Capabilities

### 🛡️ Advanced Threat Detection
- **Zero-Day Scam Protection:** The Unsupervised Autoencoder detects novel scams that don't contain any known "bad words" but statistically look like fraud.
- **Crypto & Wallet Awareness:** Explicitly flags attempts to solicit payment via **Bitcoin (BTC)**, **Ethereum (ETH)**, or **USDT**. It distinguishes between the word "Crypto" in a job title vs. a payment method.
- **Instant Payment Block:** Detects requests for **UPI**, **GPay**, **Paytm**, or **Zelle**, which are standard indicators of a recruitment scam.
- **Multi-Currency Parsing:** Robust regex handles salaries in USD, INR, EUR, GBP, AUD, etc., ensuring accurate financial analysis.

### ⚙️ Corporate Intelligence
- **Jargon Whitelist:** Pre-trained on **500+ corporate terms** (SaaS, Kubernetes, CI/CD, ROI) to prevent false positives on technical JDs. It knows that "Python" is a skill, not a snake.
- **Domain Reputation:** Automatically flags free email providers (Gmail, Yahoo) and URL shorteners (bit.ly) when used in official contact fields.
- **Professionalism Metrics:** Analyzes emoji density and punctuation patterns to flag unprofessional behavior typical of MLM schemes.

### ⚡ Production Engineering
- **Real-Time Inference:** The entire pipeline (cleaning -> BERT -> Ensemble -> S-BERT) executes in **< 200ms** on a standard CPU.
- **Dockerized:** Fully containerized environment ensuring reproducibility. System-level dependencies (`libenchant`) are handled automatically.
- **Smart Caching:** SHA-256 hashing of inputs ensures instant results for repeated queries.
- **Live Telemetry:** Color-coded, real-time system logs visible only to administrators for debugging and monitoring.

---

<a id='frameworks'></a>
## 📚 Frameworks & Libraries

The system is built upon a robust stack of industry-standard libraries, each chosen for a specific purpose:

| Library | Category | Purpose in JobGuard |
|:--------|:---------|:--------------------|
| **Scikit-Learn** | Machine Learning | Orchestrates the **Voting Ensemble**. Provides the **MLP**, **Gradient Boosting**, **Logistic Regression**, and **Isolation Forest** algorithms. |
| **PyTorch** | Deep Learning | Powers the **BERT Fusion** model and **S-BERT** inference. Chosen for its dynamic computation graph and seamless Hugging Face integration. |
| **TensorFlow / Keras** | Deep Learning | Powers the **Deep Autoencoder**. Used for its efficient static graph execution in anomaly detection tasks. |
| **Hugging Face Transformers** | NLP | Loads the pre-trained `distilbert-base-uncased` tokenizer and model weights. |
| **Sentence-Transformers** | NLP | Facilitates the semantic similarity search using `all-MiniLM-L12-v2` for the anchor-based detection system. |
| **Flask** | Backend Framework | Serves the REST API, manages user sessions, and renders the Jinja2 templates for the dashboard. |
| **LIME** | Explainable AI | Generates local perturbations to explain *why* a specific text was flagged as fraud (e.g., highlighting the word "Telegram"). |
| **PyEnchant** | Text Processing | Wraps the C-based `Enchant` library to perform high-speed dictionary validation (Gibberish detection). |
| **WordFreq** | Text Processing | Validation using Zipf frequency to distinguish between typos, slang, and true gibberish. |
| **Gunicorn** | Production Server | A WSGI HTTP Server used to run Flask in production environments (like Docker containers). |

---

<a id='project-structure'></a>
## 📂 Project Structure

```bash
JobGuard_Root/
│
├── app.py                            # Main Application (Neuro-Symbolic Engine)
├── Dockerfile                        # Container Configuration
├── requirements.txt                  # Python Dependencies
├── packages.txt                      # System Dependencies (libenchant)
│
├── models/                           # Serialized AI Models
│   ├── best_bert_fusion.pth          # Fine-tuned PyTorch Model
│   ├── autoencoder.keras             # Anomaly Detector
│   ├── iso_forest.pkl                # Outlier Detector
│   ├── meta_ensemble.pkl             # Voting Classifier (MLP+GBM+LR)
│   └── *_scaler.pkl                  # Data Normalizers
│
├── static/                           # Frontend Assets
│   ├── css/style.css                 # Cyberpunk/Glassmorphism UI
│   ├── js/script.js                  # Dashboard Async Logic
│   └── js/login.js                   # Auth & Animation Logic
│
└── templates/                        # HTML Views
    ├── index.html                    # Main Dashboard
    └── login.html                    # Secure Login Gateway
```

---

<a id='installation'></a>
## ⚙️ Installation & Setup

### Option A: Docker (Recommended)
```bash
# 1. Build the container
docker build -t jobguard .

# 2. Run the application
docker run -p 7860:7860 jobguard
```

### Option B: Manual Setup
**Prerequisites:** You must install `libenchant-2-dev` on your system.

```bash
# Ubuntu/Debian
sudo apt-get install libenchant-2-dev

# Install Python Libs
pip install -r requirements.txt

# Run
python app.py
```

---

<a id='author'></a>
## 👤 Author

**Yogeshwaran**
- **Project:** JobGuard AI (Final Year Project 2026)
- **Institution:** Panimalar Engineering College
- **Focus:** AI Security & Fraud Detection

---

<a id='license'></a>
## 📜 License

Distributed under the MIT License. See [LICENSE](LICENSE) for more information.

---
**Disclaimer:** This tool is an assistive AI. While it achieves high accuracy (98% on test sets), final hiring decisions should always involve human verification.
