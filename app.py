import os
import streamlit as st
import pandas as pd
from hf_model import predict_emotion, generate_response

# Initialize session state for history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# ----------------- Therapist Response Generator -----------------
def get_empathetic_response(emotion):
    responses = {
        "joy": "I'm really happy to hear that! 😊 Want to talk more about what's making you feel this way?",
        "sadness": "I'm sorry you're feeling this way. It's okay to feel sad. Do you want to talk about it?",
        "anger": "It seems like something upset you. I’m here to listen if you want to vent.",
        "fear": "That sounds scary. You’re not alone—would you like to share more?",
        "love": "That's beautiful. Love is powerful. How are you feeling right now?",
        "surprise": "Wow! That’s unexpected. Want to tell me more?",
        "neutral": "Thanks for sharing. I'm here if you want to talk more.",
        "others": "I'm here to support you no matter what you're feeling.",
        "happy": "It's wonderful to hear that you're feeling happy. Keep embracing the positive moments!",
        "sad": "I'm sorry you're feeling down. It's okay to feel sad—try to be gentle with yourself.",
        "angry": "Anger is a valid emotion. Take a deep breath—processing your feelings calmly is powerful.",
        "disgust": "That sounds uncomfortable. Consider distancing yourself from what’s triggering that feeling.",
        "calm": "You seem to be feeling calm. That's a wonderful state—try to hold onto it."
    }
    return responses.get(emotion.lower(), "I'm here to support you.")


# ----------------- Streamlit Config -----------------
st.set_page_config(
    page_title="NeuroAid | AI Mental Health Companion",
    page_icon="🧠",
    layout="wide"
)

# ----------------- Sidebar Branding -----------------
with st.sidebar:
    st.markdown("""
        <h2 style='color: #6C63FF;'>🧠 NeuroAid</h2>
        <p style='font-size: 0.95em;'>AI Mental Health Assistant</p>
        <hr>
    """, unsafe_allow_html=True)
    st.info("Navigate through the app and explore emotional tone analysis + therapist-style support.")
    st.markdown("💡 Created by Team NeuroAid")

# ----------------- Header -----------------
st.markdown("""
    <div style='text-align: center; padding: 20px 0;'>
        <h1 style='color: #6C63FF; font-size: 3em;'>🧠 NeuroAid</h1>
        <p style='font-size: 1.2em; color: #4F4F4F;'>Your AI-Powered Mental Health Companion</p>
        <p style='font-size: 1.1em; color: #707070;'>Understand your emotional tone and get supportive, therapist-style responses.</p>
    </div>
    <hr style='border: 1px solid #eee;' />
""", unsafe_allow_html=True)

# ----------------- Dataset Preview -----------------
DATA_FOLDER = 'data'
SUPPORTED_EXT = ['.csv', '.txt']


def load_datasets():
    datasets = []
    for file in os.listdir(DATA_FOLDER):
        if any(file.endswith(ext) for ext in SUPPORTED_EXT):
            try:
                path = os.path.join(DATA_FOLDER, file)
                if file.endswith('.csv'):
                    df = pd.read_csv(path)
                elif file.endswith('.txt'):
                    df = pd.read_csv(path, sep='\t', names=['text', 'emotion'], encoding='utf-8', engine='python')
                else:
                    continue

                df.columns = [col.lower().strip() for col in df.columns]
                if 'sentence' in df.columns:
                    df.rename(columns={'sentence': 'text'}, inplace=True)
                if 'label' in df.columns:
                    df.rename(columns={'label': 'emotion'}, inplace=True)
                if 'emotionlabel' in df.columns:
                    df.rename(columns={'emotionlabel': 'emotion'}, inplace=True)

                datasets.append((file, df))
            except Exception as e:
                st.error(f"Error loading {file}: {e}")
    return datasets


datasets = load_datasets()

if not datasets:
    st.error("No datasets found. Please add your data in the 'data/' folder.")
    st.stop()

selected_dataset = st.selectbox("📁 Choose a Dataset", [name for name, _ in datasets])
df = dict(datasets)[selected_dataset]

with st.expander("📊 Preview Selected Dataset", expanded=False):
    st.markdown("**Dataset Columns:**")
    st.code(", ".join(df.columns))
    st.dataframe(df.head(), use_container_width=True)

with st.container():
    st.markdown("""
        <div style='background-color: #f0f4ff; padding: 1em; border-left: 5px solid #6C63FF; margin-bottom: 1em; border-radius: 5px;'>
            <h4 style='color: #3A6BFF;'>👋 Welcome to NeuroAid!</h4>
            <p style='color: #555;'>Your mental health matters. Just start typing your thoughts, and let our AI help you understand and manage your emotions — with empathy, care, and science.</p>
        </div>
    """, unsafe_allow_html=True)

# ----------------- User Input -----------------
st.subheader("📝 Type or Paste Text for Analysis")
user_input = st.text_area("Type something about how you're feeling...", "", height=150)

if st.button("🔍 Analyze Emotion"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        # 🎯 Emotion Prediction via HuggingFace
        emotion_label, scores = predict_emotion(user_input)

        # 🩺 AI Therapist-style response
        response = generate_response(emotion_label)

        # 💾 Save to session history
        st.session_state.chat_history.append({
            "input": user_input,
            "emotion": emotion_label,
            "response": response
        })

        # 🎉 Display Prediction
        # Display: Emotion Label
        st.markdown(f"""
            <div style='padding: 1em; background-color: #fff3cd; border-left: 5px solid #f0ad4e; margin-bottom: 1em; border-radius: 6px;'>
                <strong>🧠 Predicted Emotion:</strong> <span style='color:#f0ad4e;'>{emotion_label}</span>
            </div>
        """, unsafe_allow_html=True)

        # Display: AI Therapist Response (Chat Bubble style)
        st.markdown(f"""
            <div style='background: #eaf3ff; padding: 1em 1.5em; border-radius: 10px; border-left: 6px solid #6C63FF; font-size: 1rem; margin-bottom: 1em;'>
                <strong>🩺 NeuroAid says:</strong><br>{response}
            </div>
        """, unsafe_allow_html=True)

        # 📊 Confidence Scores & Chart
        with st.expander("📊 Detailed Emotion Confidence Scores"):
            for item in scores:
                st.write(f"{item['label'].capitalize()}: {round(item['score'] * 100, 2)}%")

            # Horizontal bar chart
            import matplotlib.pyplot as plt

            labels = [item['label'].capitalize() for item in scores]
            confidences = [item['score'] for item in scores]

            fig, ax = plt.subplots()
            ax.barh(labels, confidences, color='#6C63FF')
            ax.set_xlim(0, 1)
            ax.set_xlabel('Confidence')
            ax.set_title('Model Confidence for Each Emotion')
            st.pyplot(fig)

# ----------------- Conversation History -----------------
with st.expander("🧾 View Your Session History", expanded=False):
    if not st.session_state.chat_history:
        st.info("No conversation history yet.")
    else:
        for i, chat in enumerate(reversed(st.session_state.chat_history), 1):
            st.markdown(f"""
            <div style='margin-bottom: 1em; padding: 1em; background-color: #F9F9F9; border-left: 4px solid #6C63FF;'>
                <strong>🗣️ You:</strong> {chat["input"]}<br>
                <strong>🔍 Emotion:</strong> <span style='color:#6C63FF;'>{chat["emotion"]}</span><br>
                <strong>🩺 NeuroAid:</strong> {chat["response"]}
            </div>
            """, unsafe_allow_html=True)

if st.button("🧹 Clear History"):
    st.session_state.chat_history.clear()
    st.success("History cleared.")
# ----------------- Footer -----------------
st.markdown("""
<hr>
<div style='text-align: center; color: #888; font-size: 13px; margin-top: 2em;'>
    NeuroAid v1.1 | Built with ❤️ using Streamlit & HuggingFace | All rights reserved © 2025
</div>
""", unsafe_allow_html=True)
