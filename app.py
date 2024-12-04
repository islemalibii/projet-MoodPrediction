from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
import pickle

# Load the saved model and preprocessing artifacts
model = tf.keras.models.load_model("mood_predictor.h5")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Flask app setup
app = Flask(__name__)

# Route for the home page
@app.route("/")
def home():
    return render_template("index.html")  # Create an HTML file for the interface

# Route for prediction
@app.route("/predict", methods=["POST"])
def predict():
    # Extract input data from the form
    song_lyrics = request.form.get("lyrics", "")
    danceability = float(request.form.get("danceability", 0))
    energy = float(request.form.get("energy", 0))
    valence = float(request.form.get("valence", 0))
    tempo = float(request.form.get("tempo", 0))
    acousticness = float(request.form.get("acousticness", 0))
    loudness = float(request.form.get("loudness", 0))
    liveness = float(request.form.get("liveness", 0))

    # Preprocess numerical features
    numerical_features = np.array([[danceability, energy, valence, tempo, acousticness, loudness, liveness]])
    numerical_features_scaled = scaler.transform(numerical_features)

    # Preprocess lyrics
    lyrics_sequence = tokenizer.texts_to_sequences([song_lyrics])
    padded_lyrics = tf.keras.utils.pad_sequences(lyrics_sequence, maxlen=250, truncating="post")

    # Predict mood
    prediction = model.predict([numerical_features_scaled, padded_lyrics])
    predicted_class = np.argmax(prediction, axis=1)[0]
    mood_label = label_encoder.inverse_transform([predicted_class])[0]

    # Return prediction result
    return jsonify({
        "predicted_mood": mood_label,
        "confidence": float(np.max(prediction))
    })

if __name__ == "__main__":
    app.run(debug=True)
