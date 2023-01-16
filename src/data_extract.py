import pandas as pd
import sqlite3

def extract_data(x):
    global df
    # Create a SQL connection to our SQLite database
    con = sqlite3.connect(x)

    df = pd.read_sql_query("SELECT * FROM failure", con)
    
    con.close() 
    return df

