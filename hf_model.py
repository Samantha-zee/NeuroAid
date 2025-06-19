from transformers import pipeline

# Load HuggingFace model pipeline
emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)

def predict_emotion(text):
    results = emotion_classifier(text)[0]
    results_sorted = sorted(results, key=lambda x: x['score'], reverse=True)
    return results_sorted[0]['label'], results_sorted

def generate_response(emotion):
    responses = {
        "joy": "I'm really happy to hear that! 😊 Want to talk more about what's making you feel this way?",
        "sadness": "I'm sorry you're feeling this way. It's okay to feel sad. Do you want to talk about it?",
        "anger": "It seems like something upset you. I’m here to listen if you want to vent.",
        "fear": "That sounds scary. You’re not alone—would you like to share more?",
        "love": "That's beautiful. Love is powerful. How are you feeling right now?",
        "surprise": "Wow! That’s unexpected. Want to tell me more?",
        "neutral": "Thanks for sharing. I'm here if you want to talk more.",
    }
    return responses.get(emotion.lower(), "I'm here to support you.")
