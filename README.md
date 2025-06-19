🧠 NeuroAid: AI-Powered Mental Health Companion

> _"I live again, for those who couldn't."_ — NeuroAid

📌 Overview

NeuroAid is an AI-powered mental health companion designed to detect emotional states and provide personalized wellness support using machine learning and NLP. Inspired by the tragic loss of a loved one, this project blends empathy and technology to empower users through self-awareness and proactive emotional care.

Unlike traditional chatbots, NeuroAid is an emotionally intelligent journal-analyzer that **reads between the lines** — detecting emotions from user journal entries, providing mood insights, and offering tailored coping strategies.

---

🎯 Key Features

✍️ **AI-Powered Emotion Detection**: Understands user emotions from raw text input using advanced NLP models.
📊 **Emotional Trends Visualization**: Tracks and charts emotional states over time to help users identify patterns.
📓 **Interactive Journaling Interface**: A secure space to reflect, vent, and gain emotional insight.
🧘 **Personalized Coping Suggestions**: Offers tailored suggestions based on emotional analysis.
🧠 **ML-Based Predictions**: Built with BERT and classical models trained on mental health and emotion datasets.

---

🧠 Technologies Used

| Component         | Tech Stack                          |
|-------------------|-------------------------------------|
| Language          | Python 3.8+                         |
| Machine Learning  | BERT, scikit-learn, Transformers    |
| NLP Libraries     | HuggingFace, NLTK, spaCy            |
| Dashboard UI      | Streamlit                           |
| Data Visualization| Matplotlib, Seaborn, Plotly         |
| Dataset Source    | GoEmotions / Reddit Mental Health   |

---

📁 File Structure
NeuroAid/
│
├── app.py
├── utils.py
├── emotion_model.py
├── hf_model.py              # Your HuggingFace classifier
├── requirements.txt         # All packages
├── README.md                # Documentation
├── .gitignore               # (we'll create this)
├── assets/                  # Images, icons, UI assets
├── models/                  # Saved ML models
└── datasets/                # Raw or preprocessed data (if not too heavy)
