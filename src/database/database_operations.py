import sqlite3
import pandas as pd

def save_to_sqlite(data, db_path, table_name):
    conn = sqlite3.connect(db_path)
    data.to_sql(table_name, conn, if_exists='replace', index=False)
    conn.close()
