from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the conversational model
model = load_model("text_similarity_model.h5")

# Preprocess and vectorize the dataset
def preprocess_text(text):
    return text.lower()

df = pd.read_csv("NewDataset.csv")
df['Processed_Question'] = df['Question'].apply(preprocess_text)
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['Processed_Question'])

def get_answer(question):
    # Preprocess user input
    processed_input = preprocess_text(question)

    # Vectorize user input
    user_vector = vectorizer.transform([processed_input])

    # Calculate cosine similarity between user input and dataset
    similarities = cosine_similarity(user_vector, tfidf_matrix).flatten()

    # Get the index of the most similar question
    most_similar_index = similarities.argmax()

    # Return the corresponding answer
    answer = df.iloc[most_similar_index]['Answer']
    return answer

@app.route("/chatbot", methods=["POST"])
def index():
    if request.method == "POST":
        question = request.form.get('Question')
        print(f"Received question: {question}")  # Add this line for debugging

        if not question:
            return jsonify({"error": "true", "message": "Please provide a question."})

        try:
            # Get the answer based on similarity
            bot_response = get_answer(question)
            
            # Prepare the response data
            data = {
                "error": "false",
                "message": "success",
                "botResponse": bot_response
            }
            return jsonify(data)
        except Exception as e:
            return jsonify({"error": "true", "message": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
