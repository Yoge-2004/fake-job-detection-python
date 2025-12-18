# Fake Job Description Detection System

![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.6.1-orange?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Status](https://img.shields.io/badge/Status-Active-green?style=for-the-badge)

A Machine Learning project designed to identify fraudulent job postings and recruitment scams. This system utilizes Natural Language Processing (NLP) to analyze job descriptions and classify them as 'Real' or 'Fake' (Scam).

## 📋 Table of Contents
- [🧐 Overview](#-overview)
- [🛠 Tech Stack](#-tech-stack)
- [⚙️ Installation](#-installation)
- [🚀 Usage](#-usage)
- [📂 File Structure](#-file-structure)
- [👤 Author](#-author)

## 🧐 Overview
Recruitment scams are becoming increasingly sophisticated. This project leverages the power of **spaCy** for text processing and **Scikit-Learn** for classification to screen job descriptions.

**Key Features:**
* **Text Preprocessing:** Efficient tokenization and cleaning using spaCy.
* **ML Classification:** Predicts probability of a job being a scam.
* **Model Persistence:** Saves and loads trained models using Joblib for quick inference.

## 🛠 Tech Stack
* **Language:** Python 3.12.12
* **Machine Learning:** Scikit-Learn 1.6.1
* **NLP:** spaCy 3.8.11
* **Math/Arrays:** NumPy 2.0.2
* **Serialization:** Joblib 1.5.2

## ⚙️ Installation

**1. Clone the repository**
```bash
git clone [https://github.com/Yoge-2004/Fake_Job_Detection_Python.git](https://github.com/Yoge-2004/Fake_Job_Detection_Python.git)
cd Fake_Job_Detection_Python
```

**2. Create a Virtual Environment**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

**3. Install Dependencies**
```bash
pip install -r requirements.txt
```

**4. Download spaCy Language Model**
```bash
python -m spacy download en_core_web_md
```

## 🚀 Usage

**To Run the Code:**
```bash
python app.py
```

**Example Inference Code:**
```python
import joblib

# Load the model
model = joblib.load('job_fraud_model.pkl')

# Sample text
job_desc = ['Urgent hiring! No interview required. Send bank details immediately.']

# Predict
prediction = model.predict(job_desc)
print(f'Verdict: {prediction[0]}')
```

## 📂 File Structure

```text
Fake_Job_Detection_Python/
│
├── fake_job_postings.csv   # Dataset
├── production_fake_job_pipeline.pkl     # Trained model
├── results.csv           # Test Results
├── app.py                 # Main script
├── requirements.txt        # Dependencies
└── README.md               # Documentation
```

## 👤 Author
**Yoge-2004**
* GitHub: [Yoge-2004](https://github.com/Yoge-2004)

---
*Disclaimer: This tool is for educational purposes.*
