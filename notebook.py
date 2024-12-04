import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences, to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout, concatenate, GlobalAveragePooling1D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def advanced_mood_clustering(df):
    """
    Refined mood clustering with more nuanced feature analysis
    """
    def get_mood_cluster(row):
        # Passionate, Rousing, Confident, Boisterous, Rowdy (Cluster 1)
        if (row['energy'] > 0.75 and 
            row['valence'] > 0.6 and 
            row['tempo'] > 130 and 
            row['loudness'] > -3):
            return "passionate"
        
        # Rollicking, Cheerful, Fun, Sweet, Amiable (Cluster 2)
        elif (row['energy'] > 0.6 and 
              row['valence'] > 0.5 and 
              row['danceability'] > 0.6 and 
              row['tempo'] > 110):
            return "cheerful"
        
        # Literate, Poignant, Wistful, Bittersweet, Autumnal, Brooding (Cluster 3)
        elif (row['energy'] < 0.4 and 
              row['valence'] < 0.4 and 
              row['acousticness'] > 0.5 and 
              row['tempo'] < 100):
            return "melancholic"
        
        # Humorous, Silly, Campy, Quirky, Whimsical, Witty, Wry (Cluster 4)
        elif (row['danceability'] > 0.5 and 
              row['valence'] > 0.5 and 
              row['energy'] < 0.5):
            return "playful"
        
        return "neutral"

    df['mood_cluster'] = df.apply(get_mood_cluster, axis=1)
    return df

def preprocess_advanced(csv_path):
    """
    Enhanced data preprocessing with more robust feature engineering
    """
    df = pd.read_csv(csv_path)
    
    numeric_columns = [
        "danceability", "energy", "valence", "tempo", 
        "acousticness", "loudness", "liveness"
    ]
    
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].fillna(df[col].median())
    
    df = advanced_mood_clustering(df)
    
    scaler = StandardScaler()
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
    
    label_encoder = LabelEncoder()
    df['mood_encoded'] = label_encoder.fit_transform(df['mood_cluster'])
    
    return df, numeric_columns, label_encoder

def advanced_text_preprocessing(df, max_vocab=15000, max_len=250):
    """
    Improved text preprocessing with TF-IDF and Tokenization
    """
    lyrics = df['lyrics'].fillna('').astype(str)
    
    tfidf = TfidfVectorizer(max_features=max_vocab)
    text_features = tfidf.fit_transform(lyrics).toarray()
    
    tokenizer = Tokenizer(num_words=max_vocab, oov_token="<OOV>")
    tokenizer.fit_on_texts(lyrics)
    sequences = tokenizer.texts_to_sequences(lyrics)
    padded_sequences = pad_sequences(sequences, maxlen=max_len, truncating='post')
    
    return text_features, padded_sequences, tokenizer

def build_advanced_model(num_features, text_vocab_size, text_max_len, num_classes):
    """
    More sophisticated neural network architecture
    """
    num_input = Input(shape=(num_features,), name='numeric_input')
    x_num = Dense(128, activation='relu')(num_input)
    x_num = Dropout(0.4)(x_num)
    x_num = Dense(64, activation='relu')(x_num)
    
    text_input = Input(shape=(text_max_len,), name='text_input')
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

def main():
    csv_path = "songs_data_with_lyrics.csv"
    
    df, numeric_features, label_encoder = preprocess_advanced(csv_path)
    
    text_features, text_sequences, tokenizer = advanced_text_preprocessing(df)
    
    numerical_features = df[numeric_features].values
    y = df['mood_encoded'].values
    
    X_text_train, X_text_test, X_num_train, X_num_test, y_train, y_test = train_test_split(
        text_sequences, numerical_features, y, 
        test_size=0.2, random_state=42, stratify=y
    )
    
    model = build_advanced_model(
        num_features=len(numeric_features),
        text_vocab_size=len(tokenizer.word_index) + 1,
        text_max_len=250,
        num_classes=len(label_encoder.classes_)
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss', 
        patience=7, 
        restore_best_weights=True
    )
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.2, 
        patience=4, 
        min_lr=0.00001
    )
    
    history = model.fit(
        [X_num_train, X_text_train], y_train,
        validation_data=([X_num_test, X_text_test], y_test),
        epochs=75,  
        batch_size=64,  
        callbacks=[early_stopping, reduce_lr]
    )
    
    y_pred = model.predict([X_num_test, X_text_test]).argmax(axis=1)
    print("\nClassification Report:")
    print(classification_report(
        y_test, y_pred, 
        target_names=label_encoder.classes_
    ))
    
    plt.figure(figsize=(10,6))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()