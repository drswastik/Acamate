import os
import sqlite3
from config.config import DB_PATH

class DatabaseManager:
    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path
        db_dir = os.path.dirname(self.db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
            
        self.conn = None
        self.cursor = None
        self.connect()

    def connect(self):
        """Connect to SQLite database."""
        try:
            # isolation_level=None forces Auto-Commit mode. 
            # This completely stops SQLite from caching stale read-snapshots between Streamlit reruns
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False, isolation_level=None)
            self.cursor = self.conn.cursor()
            # Enable Foreign Keys
            self.cursor.execute("PRAGMA foreign_keys = ON;")
        except sqlite3.Error as e:
            print(f"❌ DB connection error: {e}")

    def create_tables(self):
        """Create all tables from schema.sql."""
        try:
            schema_path = os.path.join("db", "schema.sql")
            with open(schema_path, "r", encoding="utf-8") as f:
                schema = f.read()
            self.cursor.executescript(schema)
            self.conn.commit()
            print("🗄️ Database schema verified with Conversational Memory support.")
        except Exception as e:
            print(f"❌ Error creating tables: {e}")

    #paper operations
    def insert_paper(self, file_name: str, file_hash: str, file_path: str):
        """Insert new paper record if not present."""
        try:
            self.cursor.execute("""
                INSERT OR IGNORE INTO papers (file_name, file_hash, file_path)
                VALUES (?, ?, ?)
            """, (file_name, file_hash, file_path))
            self.conn.commit()
        except sqlite3.Error as e:
            print(f"❌ Error inserting paper: {e}")

    def get_paper_id(self, file_hash: str):
        """Fetch paper ID using its content MD5 hash."""
        self.cursor.execute("SELECT id FROM papers WHERE file_hash = ?", (file_hash,))
        row = self.cursor.fetchone()
        return row[0] if row else None

    def get_all_papers(self) -> list:
        """Fetch all previously uploaded papers for the Library UI."""
        self.cursor.execute("SELECT id, file_name, file_hash, uploaded_at FROM papers ORDER BY uploaded_at DESC")
        return self.cursor.fetchall()

    #mia operations
    def get_mia_map(self, paper_id: int):
        """Fetch cached MiA global summary for a paper."""
        self.cursor.execute("SELECT global_summary, key_concepts FROM mia_maps WHERE paper_id = ?", (paper_id,))
        row = self.cursor.fetchone()
        return {"global_summary": row[0], "key_concepts": row[1]} if row else None

    def insert_mia_map(self, paper_id: int, global_summary: str, key_concepts: str = ""):
        """Store generated MiA global summary."""
        try:
            self.cursor.execute("""
                INSERT OR REPLACE INTO mia_maps (paper_id, global_summary, key_concepts)
                VALUES (?, ?, ?)
            """, (paper_id, global_summary, key_concepts))
            self.conn.commit()
        except sqlite3.Error as e:
            print(f"❌ Error inserting MiA map: {e}")

    #chat session operations
    def create_chat_session(self, paper_id: int, session_name: str) -> int:
        """Create a new conversation session for a specific paper."""
        try:
            self.cursor.execute("""
                INSERT INTO chat_sessions (paper_id, session_name)
                VALUES (?, ?)
            """, (paper_id, session_name))
            self.conn.commit()
            return self.cursor.lastrowid
        except sqlite3.Error as e:
            print(f"❌ Error creating chat session: {e}")
            return None

    def get_sessions_for_paper(self, paper_id: int) -> list:
        """Get all past chat sessions associated with a specific paper ID."""
        self.cursor.execute("""
            SELECT id, session_name, created_at, updated_at 
            FROM chat_sessions 
            WHERE paper_id = ? 
            ORDER BY updated_at DESC
        """, (paper_id,))
        return self.cursor.fetchall()

    def add_chat_message(self, session_id: int, role: str, content: str, 
                         routing_provider: str = None, faithfulness_score: float = None):
        """Add a single message (user or assistant) to a chat session."""
        try:
            self.cursor.execute("""
                INSERT INTO chat_messages (session_id, role, content, routing_provider, faithfulness_score)
                VALUES (?, ?, ?, ?, ?)
            """, (session_id, role, content, routing_provider, faithfulness_score))
            
            self.cursor.execute("""
                UPDATE chat_sessions SET updated_at = CURRENT_TIMESTAMP WHERE id = ?
            """, (session_id,))
            
            self.conn.commit()
        except sqlite3.Error as e:
            print(f"❌ Error recording chat message: {e}")

    def get_chat_history(self, session_id: int, limit: int = 6) -> list:
        """Fetch the last N messages from a session formatted for LLM context."""
        self.cursor.execute("""
            SELECT role, content FROM (
                SELECT id, role, content FROM chat_messages 
                WHERE session_id = ? 
                ORDER BY id DESC LIMIT ?
            ) ORDER BY id ASC
        """, (session_id, limit))
        
        rows = self.cursor.fetchall()
        return [{"role": r[0], "content": r[1]} for r in rows]

    #query cache operations
    def get_cached_response(self, query_hash: str):
        """Return cached query result."""
        self.cursor.execute("""
            SELECT response, routing_provider, faithfulness_score 
            FROM query_cache WHERE query_hash = ?
        """, (query_hash,))
        row = self.cursor.fetchone()
        if row:
            return {
                "response": row[0],
                "routing_provider": row[1],
                "faithfulness_score": row[2]
            }
        return None

    def insert_query_response(self, paper_id: int, query_hash: str, user_query: str, 
                              refined_query: str, response: str, 
                              provider: str, faithfulness_score: float = None):
        """Insert query response into query_cache."""
        try:
            self.cursor.execute("""
                INSERT OR REPLACE INTO query_cache 
                (paper_id, query_hash, user_query, refined_query, response, routing_provider, faithfulness_score)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (paper_id, query_hash, user_query, refined_query, response, provider, faithfulness_score))
            self.conn.commit()
        except sqlite3.Error as e:
            print(f"❌ Error inserting query response: {e}")

    def update_feedback(self, query_hash: str, feedback_value: int):
        """Update feedback rating."""
        try:
            self.cursor.execute("""
                UPDATE query_cache SET feedback = ? WHERE query_hash = ?
            """, (feedback_value, query_hash))
            self.conn.commit()
        except sqlite3.Error as e:
            print(f"❌ Error updating feedback: {e}")

    def close(self):
        """Close DB connection."""
        if self.conn:
            self.conn.close()
