import sqlite3
import hashlib

def init_db():
    conn = sqlite3.connect("app.db")
    c = conn.cursor()

    c.execute("""
    CREATE TABLE IF NOT EXISTS history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT,
        caption TEXT,
        generated_caption TEXT,
        hashtags TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)

    c.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        password TEXT
    )
    """)

    conn.commit()
    conn.close()


def save_post(username, caption, generated_caption, hashtags):
    conn = sqlite3.connect("app.db")
    c = conn.cursor()

    c.execute("""
    INSERT INTO history (username, caption, generated_caption, hashtags)
    VALUES (?, ?, ?, ?)
    """, (username, caption, generated_caption, hashtags))

    conn.commit()
    conn.close()


def get_history(username):
    conn = sqlite3.connect("app.db")
    c = conn.cursor()

    c.execute("""
    SELECT caption, generated_caption, hashtags, timestamp 
    FROM history 
    WHERE username=?
    ORDER BY id DESC
    """, (username,))

    rows = c.fetchall()
    conn.close()

    return rows

def create_user(username, password):
    conn = sqlite3.connect("app.db")
    c = conn.cursor()

    hashed_password = hashlib.sha256(password.encode()).hexdigest()

    try:
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", 
                  (username, hashed_password))
        conn.commit()
    except:
        pass

    conn.close()


def validate_user(username, password):
    conn = sqlite3.connect("app.db")
    c = conn.cursor()

    hashed_password = hashlib.sha256(password.encode()).hexdigest()

    c.execute("SELECT * FROM users WHERE username=? AND password=?", 
              (username, hashed_password))
    
    user = c.fetchone()
    conn.close()

    return user is not None