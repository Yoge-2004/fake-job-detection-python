# Fake Job Posting and Online Recruitment Scam Detection Using Hybrid AI & NLP

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE) [![Python](https://img.shields.io/badge/Python-3.12.12-blue)](https://www.python.org/) [![Framework](https://img.shields.io/badge/Backend-Flask-red)](https://flask.palletsprojects.com/) [![ML](https://img.shields.io/badge/sklearn-1.6.1-orange)]() [![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19-orange)]()

## 📋 Table of Contents
1. [📌 Overview](#-overview)
2. [✨ Key Features (Application)](#-key-features-application)
3. [🛠️ Tech Stack & Compatibility](#-tech-stack--compatibility)
4. [🔬 Deep Dive: Model Architecture](#-deep-dive-model-architecture)
5. [📂 Project Structure](#-project-structure)
6. [⚙️ Installation & Setup](#-installation--setup)
7. [👤 Author](#-author)
8. [📜 License](#-license)

---

## 📌 Overview
This project is a sophisticated **Web-based AI Deception Detector** designed to identify fake job postings and recruitment scams. Unlike static spam filters, this system employs a **Dual-Core Hybrid Intelligence Engine**.

It combines:
1. **Supervised Learning:** To detect contextual fraud (e.g., fake promises, payment scams).
2. **Unsupervised Deep Learning:** To detect **Zero-Day Attacks** and structural anomalies (e.g., gibberish, weird formatting) using Autoencoders.
3. **Rule-based Logic:** FlashText for instant blocking of known scam keywords.

The application offers a secure, user-friendly dashboard where candidates can paste job descriptions and receive an instant **Trust Score**, **Risk Breakdown**, and **Explainable AI Insights**.

---

## ✨ Key Features (Application)

### 🛡️ Core Security & Detection
- **Dual-Core AI Engine:** Runs both a Supervised Classifier (Context) and an Unsupervised Anomaly Detector (Structure) simultaneously.
- **Zero-Day Scam Protection:** The Unsupervised model (Autoencoder) detects never-before-seen scams by flagging structural irregularities (gibberish, symbol abuse, chaotic formatting).
- **Critical Trigger Bypass (FlashText):** Instantly flags high-risk terms (e.g., *Telegram, Signal, Wire Transfer*) with 0% latency.
- **Smart Boost Logic:** If an anomaly is detected, the system automatically boosts the risk score to 'Critical' levels.

### ⚡ High-Performance Architecture
- **Smart RAM Caching:** Implements `Flask-Caching` to store analysis results. Repeated queries return results in **0.001ms**.
- **Optimized Pipeline:** Models are serialized with `joblib` and pre-loaded into memory for rapid inference.

### 🔍 Explainable AI (XAI)
- **LIME Integration:** Explains *which words* triggered the fraud score.
- **Anomaly Explanation:** Unlike standard models, our Anomaly Detector explains *why* the structure is bad (e.g., *"Too many Capital Letters", "Excessive Symbols"*).

---

## 🛠️ Tech Stack & Compatibility

**Core Environment:**
- **Python:** 3.12.12 (Strict Requirement)
- **Scikit-Learn:** 1.6.1
- **TensorFlow / Keras:** 2.19 (For Deep Learning)

**Backend & Logic:**
- **Framework:** Flask (Python)
- **NLP Engine:** Spacy (`en_core_web_lg`), FlashText, Regular Expressions
- **Deep Learning:** Autoencoders (Unsupervised Anomaly Detection)
- **Interpretation:** LIME (Local Interpretable Model-agnostic Explanations)

**Frontend:**
- HTML5, CSS3 (Custom Animations), JavaScript (Fetch API)

---

## 🔬 Deep Dive: Model Architecture
This system uses a novel **Dual-Pipeline Approach**, running two independent AI models in parallel to ensure 360° protection.

### 🧠 Pipeline 1: Supervised Context Awareness
*Objective: To understand WHAT is written (Semantic Meaning).* 

1. **Data Preprocessing:**
   - **Spacy NLP:** Utilizes `en_core_web_lg` (Large English Model) to convert text into 300-dimensional semantic vectors.
   - **SMOTE (Synthetic Minority Over-sampling Technique):** Solves the class imbalance problem by generating synthetic examples of fake jobs during training, ensuring the model isn't biased toward real jobs.

2. **Voting Ensemble Classifier:**
   Instead of relying on a single algorithm, we aggregate predictions from three robust models:
   - **Multi-Layer Perceptron (MLP):** A Neural Network that captures non-linear relationships in text data.
   - **Gradient Boosting:** Excellent for handling tabular nuances and decision boundaries.
   - **Logistic Regression:** Provides a solid baseline probability score.
   *Result:* The final output is a weighted probability (Soft Voting) of the input being a scam.

### 🦄 Pipeline 2: Unsupervised Structural Anomaly Detection
*Objective: To analyze HOW it is written (Syntax & Structure). Detects 'Zero-Day' scams that use new words but suspicious patterns.*

1. **Feature Engineering (307 Dimensions):**
   - **300 Dim:** Spacy Semantic Vectors.
   - **7 Dim (Structural):** Calculated ratios of Capital Letters, Digits, Special Characters, URL count, Email count, and Word Density.

2. **Deep Autoencoder (Reconstruction Logic):**
   - **Architecture:** `Input(307) -> Dense(256) -> Dense(128) -> Bottleneck(32) -> Dense(128) -> Dense(256) -> Output(307)`.
   - **Mechanism:** The Autoencoder is trained ONLY on 'Real' job descriptions. It learns to compress and reconstruct normal text perfectly.
   - **Detection:** When a 'Fake' or 'Gibberish' text is fed in, the model fails to reconstruct it accurately. The resulting **Reconstruction Error (MSE)** is used as the Anomaly Score.

3. **Isolation Forest (Statistical Filter):**
   - Acts as a secondary filter to catch outliers based on feature distribution (e.g., text that is statistically too short or too repetitive).

---

## 📂 Project Structure
```bash
Fake_Job_Detection_Python/
│
├── app.py                            # Main Flask Application (The Engine)
├── test.py                           # Utility script to test model accuracy
├── requirements.txt                  # Dependencies (Strict Versions)
├── README.md                         # Documentation
├── LICENSE                           # MIT License File
│
├── users.db                          # User Credentials Database (Auto-generated)
├── fake_job_postings.csv             # Raw Dataset for Training
├── production_fake_job_pipeline.pkl  # Supervised AI Model
├── robust_anomaly_model.pkl          # Unsupervised Deep Learning Model
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

## ⚙️ Installation & Setup

**⚠️ Critical Note:** Ensure you have **TensorFlow 2.19** installed alongside standard ML libraries.

**1. Clone the Repository**
```bash
git clone [https://github.com/your-username/JobGuard-AI.git](https://github.com/your-username/JobGuard-AI.git)
cd JobGuard-AI
```

**2. Install Dependencies**
```bash
pip install -r requirements.txt
```

**3. Download Spacy Model**
```bash
python -m spacy download en_core_web_lg
```

**4. Run the App**
```bash
python app.py
```
Visit `http://127.0.0.1:5000` in your browser.

---

## 👤 Author

**Yogeshwaran**
- AI & Full Stack Developer
- Project: JobGuard AI (2025)

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.

---
Made with ❤️ and Python.
