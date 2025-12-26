# Fake Job Posting and Online Recruitment Scam Detection Using Machine Learning and Natural Language Processing

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE) [![Python](https://img.shields.io/badge/Python-3.12.12-blue)](https://www.python.org/) [![Framework](https://img.shields.io/badge/Backend-Flask-red)](https://flask.palletsprojects.com/) [![ML](https://img.shields.io/badge/sklearn-1.6.1-orange)]()

## 📋 Table of Contents
1. [📌 Overview](#-overview)
2. [✨ Key Features (Application)](#-key-features-application)
3. [🛠️ Tech Stack & Compatibility](#️-tech-stack--compatibility)
4. [🔬 Model Architecture (Training)](#-model-architecture-training)
5. [📂 Project Structure](#-project-structure)
6. [⚙️ Installation & Setup](#️-installation--setup)
7. [👤 Author](#-author)
8. [📜 License](#-license)

---

## 📌 Overview
This project is a sophisticated **Web-based AI Deception Detector** designed to identify fake job postings and recruitment scams. Unlike static spam filters, this system employs a **Hybrid Intelligence Engine** that combines Deep Learning (Spacy Vectors), Statistical Analysis (TF-IDF), and Rule-based Logic (FlashText) to analyze job descriptions in real-time.

The application offers a secure, user-friendly dashboard where candidates can paste job descriptions and receive an instant **Trust Score**, **Risk Breakdown**, and **Explainable AI Insights**.

---

## ✨ Key Features (Application)

### 🛡️ Core Security & Detection
- **Hybrid Threat Detection:** Seamlessly blends Machine Learning predictions with hardcoded critical triggers.
- **Critical Trigger Bypass (FlashText):** Instantly flags high-risk terms (e.g., *Telegram, Signal, Wire Transfer*) with 0% latency, bypassing the ML model for known scams.
- **Gibberish & Spam Filter:** Automatically detects and rejects nonsense input (e.g., 'asdfgh', repeated characters) before processing.
- **Phishing Link Detection:** Identifies attempts to harvest user data via 'Update Profile' or suspicious external links.

### ⚡ High-Performance Architecture
- **Smart RAM Caching:** Implements `Flask-Caching` to store analysis results. Repeated queries for the same job description return results in **0.001ms**.
- **Optimized Pipeline:** The ML model is serialized with `joblib` and pre-loaded into memory for rapid inference.

### 🔍 Explainable AI (XAI)
- **LIME Integration:** If a job is flagged as suspicious, the system explains *why* by highlighting the exact words contributing to the fraud score.
- **Smart Nuance Hiding:** Automatically hides LIME details for 'Safe' jobs to prevent user confusion, only showing insights when risks are detected.

### 💻 User Interface & Experience
- **Cyberpunk / Glassmorphism UI:** A modern, visually engaging interface with liquid animations and responsive design.
- **Real-time Admin Logs:** The dashboard displays live server logs (Spacy stats, FlashText hits) for Admin users.
- **Secure Authentication:** Complete Login/Signup system using `Werkzeug` security hashing and SQLite.

---

## 🛠️ Tech Stack & Compatibility

**Core Environment:**
- **Python:** 3.12.12 (Strict Requirement)
- **Scikit-Learn:** 1.6.1

**Backend & Logic:**
- **Framework:** Flask (Python)
- **NLP Engine:** Spacy (`en_core_web_lg`), FlashText, Regular Expressions
- **ML Utilities:** Joblib, Imbalanced-Learn, Numpy
- **Interpretation:** LIME (Local Interpretable Model-agnostic Explanations)

**Frontend:**
- HTML5, CSS3 (Custom Animations), JavaScript (Fetch API)

---

## 🔬 Model Architecture (Training)
The AI Brain (`production_fake_job_pipeline.pkl`) is not just a simple classifier. It uses a Voting Ensemble approach trained on `fake_job_postings.csv`:

1.  **Preprocessing:** HTML cleaning, URL tokenization, and character normalization.
2.  **Feature Extraction:**
    - *Semantic Vectors:* Uses Spacy to understand word context.
    - *TF-IDF N-Grams:* Captures specific fraud phrases.
3.  **Class Balancing:** Trained using **SMOTE** to handle the scarcity of scam samples.
4.  **Voting Classifier:** A weighted decision made by:
    - *Neural Network (MLP)*
    - *Gradient Boosting*
    - *Logistic Regression*

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
├── results.csv                       # Analysis Output Logs (Optional)
├── production_fake_job_pipeline.pkl  # The Trained AI Brain
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

**⚠️ Critical Note:** Ensure you are using **Python 3.12.12** and **scikit-learn 1.6.1**.

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
