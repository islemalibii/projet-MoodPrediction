# File path: preprocessing/fetch_lyrics_genius.py

import lyricsgenius
import pandas as pd
import time

# Initialize Genius API
GENIUS_ACCESS_TOKEN = "mW94GmlujP-96RaLvGBrZVVWYzKOU1eafP0U07Co_bIi73jpdYqVysXbN6Bui_V8"  
genius = lyricsgenius.Genius(GENIUS_ACCESS_TOKEN)

def fetch_lyrics_genius(artist, title):
    """
    Fetch lyrics for a given artist and title using Genius API.
    """
    try:
        song = genius.search_song(title, artist)
        return song.lyrics if song else None
    except Exception as e:
        print(f"Error fetching lyrics for {artist} - {title}: {e}")
        return None

def augment_csv_with_lyrics(input_csv, output_csv):
    """
    Add a lyrics column to the CSV by fetching lyrics for each song.
    """
    df = pd.read_csv(input_csv)
    
    # Add a new column for lyrics
    df['lyrics'] = None

    for idx, row in df.iterrows():
        artist = row['Artist']
        title = row['TrackName']
        print(f"Fetching lyrics for: {artist} - {title} ({idx + 1}/{len(df)})")
        
        lyrics = fetch_lyrics_genius(artist, title)
        df.at[idx, 'lyrics'] = lyrics

        # Avoid getting rate-limited
        time.sleep(1)

    # Save augmented data to a new CSV
    df.to_csv(output_csv, index=False)
    print(f"Lyrics added and saved to {output_csv}")

# Usage
input_csv = "songs_data.csv"
output_csv = "songs_data_with_lyrics.csv"
augment_csv_with_lyrics(input_csv, output_csv)
