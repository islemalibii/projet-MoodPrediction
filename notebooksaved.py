import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout, concatenate, GlobalAveragePooling1D, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import classification_report


MAX_VOCAB = 15000
MAX_LEN = 250
NUM_EPOCHS = 50
BATCH_SIZE = 64

def advanced_mood_clustering(df):
    """
    Generate 'mood_cluster' column by clustering based on numerical features.
    """
    def get_mood_cluster(row):
        if (row['energy'] > 0.75 and 
            row['valence'] > 0.6 and 
            row['tempo'] > 130 and 
            row['loudness'] > -3):
            return "passionate"
        elif (row['energy'] > 0.6 and 
              row['valence'] > 0.5 and 
              row['danceability'] > 0.6 and 
              row['tempo'] > 110):
            return "cheerful"
        elif (row['energy'] < 0.4 and 
              row['valence'] < 0.4 and 
              row['acousticness'] > 0.5 and 
              row['tempo'] < 100):
            return "melancholic"
        elif (row['danceability'] > 0.5 and 
              row['valence'] > 0.5 and 
              row['energy'] < 0.5):
            return "playful"
        return "neutral"


    df['mood_cluster'] = df.apply(get_mood_cluster, axis=1)
    return df



def preprocess_data(csv_path):
    """
    Preprocess the dataset, including scaling and encoding.
    """
    df = pd.read_csv(csv_path)


    numeric_columns = [
        "danceability", "energy", "valence", "tempo", 
        "acousticness", "loudness", "liveness"
    ]
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')
    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())

    if 'mood_cluster' not in df.columns:
        df = advanced_mood_clustering(df)


    scaler = StandardScaler()
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

    label_encoder = LabelEncoder()
    df['mood_encoded'] = label_encoder.fit_transform(df['mood_cluster'])

    return df, numeric_columns, label_encoder, scaler


def preprocess_text(df):
    """
    Preprocess the lyrics data using tokenization and padding.
    """
    tokenizer = Tokenizer(num_words=MAX_VOCAB, oov_token="<OOV>")
    tokenizer.fit_on_texts(df['lyrics'].fillna(''))
    sequences = tokenizer.texts_to_sequences(df['lyrics'].fillna(''))
    padded_sequences = pad_sequences(sequences, maxlen=MAX_LEN, truncating='post')
    return padded_sequences, tokenizer

def build_model(num_features, text_vocab_size, num_classes):
    """
    Build the multi-input model for mood prediction.
    """

    num_input = Input(shape=(num_features,), name='numeric_input')
    x_num = Dense(128, activation='relu')(num_input)
    x_num = BatchNormalization()(x_num)
    x_num = Dropout(0.4)(x_num)
    x_num = Dense(64, activation='relu')(x_num)


    text_input = Input(shape=(MAX_LEN,), name='text_input')
    x_text = Embedding(input_dim=text_vocab_size, output_dim=100)(text_input)
    x_text = LSTM(160, return_sequences=True)(x_text)
    x_text = GlobalAveragePooling1D()(x_text)
    x_text = Dropout(0.4)(x_text)


    combined = concatenate([x_num, x_text])
    x = Dense(96, activation='relu')(combined)
    x = Dropout(0.5)(x)
    x = Dense(48, activation='relu')(x)
    output = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=[num_input, text_input], outputs=output)
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def save_artifacts(model, tokenizer, label_encoder, scaler):
    """
    Save the trained model and preprocessing artifacts.
    """
    model.save("mood_predictor.h5")
    with open("tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)
    with open("label_encoder.pkl", "wb") as f:
        pickle.dump(label_encoder, f)
    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

def main():
    csv_path = "songs_data_with_lyrics.csv"

    df, numeric_features, label_encoder, scaler = preprocess_data(csv_path)
    text_sequences, tokenizer = preprocess_text(df)
    X_num = df[numeric_features].values
    y = df['mood_encoded'].values


    X_text_train, X_text_test, X_num_train, X_num_test, y_train, y_test = train_test_split(
        text_sequences, X_num, y, test_size=0.2, random_state=42, stratify=y
    )


    model = build_model(
        num_features=len(numeric_features),
        text_vocab_size=MAX_VOCAB,
        num_classes=len(label_encoder.classes_)
    )


    early_stopping = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=4, min_lr=0.00001)

    history = model.fit(
        [X_num_train, X_text_train], y_train,
        validation_data=([X_num_test, X_text_test], y_test),
        epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stopping, reduce_lr]
    )


    save_artifacts(model, tokenizer, label_encoder, scaler)


    y_pred = model.predict([X_num_test, X_text_test]).argmax(axis=1)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))


    plt.figure(figsize=(10, 6))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
