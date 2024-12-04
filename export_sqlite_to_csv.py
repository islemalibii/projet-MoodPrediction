import sqlite3
import pandas as pd

def export_sqlite_to_csv(db_path, table_name, output_csv):
    """
    Export a SQLite table to a CSV file.
    """
    # Connect to SQLite database
    conn = sqlite3.connect(db_path)
    
    # Read table into DataFrame
    query = f"SELECT * FROM {table_name}"
    df = pd.read_sql_query(query, conn)
    
    # Close connection
    conn.close()
    
    # Save to CSV
    df.to_csv(output_csv, index=False)
    print(f"Exported {table_name} to {output_csv}")

# Usage
db_path = "features.sqlite"
table_name = "features"
output_csv = "songs_data.csv"
export_sqlite_to_csv(db_path, table_name, output_csv)
