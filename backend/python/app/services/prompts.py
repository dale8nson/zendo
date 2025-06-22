from .db import get_connection

async def prompts():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, text, source, is_active, created_at FROM prompts WHERE is_active = 1")
    results = cursor.fetchall()
    conn.close()
    return [{"id": r[0], "text": r[1], "source": r[2], "is_active": bool(r[3]), "created_at": r[4]} for r in results]

async def add(text: str, source: str = "manual"):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO prompts (text, source) VALUES (?, ?)", (text, source))
    conn.commit()
    conn.close()
    return {"status": "ok", "text": text}

async def delete(prompt_id: int):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM prompts WHERE id = ?", (prompt_id,))
    conn.commit()
    conn.close()
    return {"status": "deleted"}

async def update(id: int, text: str, source: str):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("UPDATE prompts SET text = ?, source = ?, updated_at = datetime('now') WHERE id = ?", (text, source, id))
    conn.commit()
    conn.close()
    return { "status": f"updated prompt with id {id} to {text}"}
