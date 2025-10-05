import sqlite3
from datetime import datetime

class Database:
    def __init__(self, db_path="database.db"):
        self.db_path = db_path
        self.create_table()

    def connect(self):
        return sqlite3.connect(self.db_path)

    def create_table(self):
        conn = self.connect()
        cur = conn.cursor()
        cur.execute('''
            CREATE TABLE IF NOT EXISTS uploads (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file TEXT,
                created_at TEXT,
                updated_at TEXT
            )
        ''')
        conn.commit()
        conn.close()

    def insert_upload(self, filename: str):
        conn = self.connect()
        cur = conn.cursor()
        now = datetime.now().isoformat(sep=" ", timespec="seconds")
        cur.execute(
            "INSERT INTO uploads (file, created_at, updated_at) VALUES (?, ?, ?)",
            (filename, now, now)
        )
        conn.commit()
        conn.close()
    def get_filename_by_id(self, id):
        conn = self.connect()
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT file FROM uploads WHERE id = ?", (id,))
            result = cursor.fetchone()
            if result:
                return result[0]  # El nombre del archivo
            return None
        except Exception as e:
            print(f"Error al obtener el filename: {e}")
            return None
        finally:
            cursor.close()
            conn.close()
       