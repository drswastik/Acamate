-- =====================================================
-- AcaMate Database Schema (v4 - Conversational Memory)
-- =====================================================

-- 🧠 Table 1: Papers Metadata (Hash-based identification)
CREATE TABLE IF NOT EXISTS papers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_name TEXT NOT NULL,
    file_hash TEXT UNIQUE NOT NULL,
    file_path TEXT,
    uploaded_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- 🗺️ Table 2: MiA Global Mindscape Maps
CREATE TABLE IF NOT EXISTS mia_maps (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    paper_id INTEGER UNIQUE NOT NULL,
    global_summary TEXT NOT NULL,
    key_concepts TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (paper_id) REFERENCES papers(id)
);

-- 💬 Table 3: Chat Sessions (Per-PDF Conversations)
CREATE TABLE IF NOT EXISTS chat_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    paper_id INTEGER NOT NULL,
    session_name TEXT NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (paper_id) REFERENCES papers(id)
);

-- 🗨️ Table 4: Chat Messages (Multi-turn Memory)
CREATE TABLE IF NOT EXISTS chat_messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL,
    role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
    content TEXT NOT NULL,
    routing_provider TEXT,
    faithfulness_score REAL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES chat_sessions(id) ON DELETE CASCADE
);

-- ⚡ Table 5: Exact Query Cache
CREATE TABLE IF NOT EXISTS query_cache (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    paper_id INTEGER NOT NULL,
    query_hash TEXT UNIQUE NOT NULL,
    user_query TEXT NOT NULL,
    refined_query TEXT NOT NULL,
    response TEXT NOT NULL,
    routing_provider TEXT,
    faithfulness_score REAL,
    feedback INTEGER DEFAULT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (paper_id) REFERENCES papers(id)
);