# File path: analysis/visualize_features.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_feature_distributions(df, features):
    """
    Plot the distribution of numerical features.
    """
    for feature in features:
        plt.figure(figsize=(10, 5))
        sns.histplot(df[feature], kde=True, bins=30)
        plt.title(f"Distribution of {feature}")
        plt.xlabel(feature)
        plt.ylabel("Frequency")
        plt.show()


def plot_correlation_heatmap(df, features):
    """
    Plot a heatmap showing correlations between numerical features.
    """
    corr_matrix = df[features].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Feature Correlation Heatmap")
    plt.show()


if __name__ == "__main__":
    csv_path = "songs_data_with_lyrics.csv"
    df = pd.read_csv(csv_path)

    numerical_features = [
        "danceability", "energy", "valence", "tempo", "acousticness",
        "loudness", "liveness"
    ]


    plot_correlation_heatmap(df, numerical_features)
