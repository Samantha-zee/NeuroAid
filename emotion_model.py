from transformers import pipeline

# Load pre-trained emotion detection model
emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)

def predict_emotion(text):
    results = emotion_classifier(text)
    sorted_results = sorted(results[0], key=lambda x: x['score'], reverse=True)
    top_emotion = sorted_results[0]['label']
    return top_emotion, sorted_results
