import sqlite3

DB_PATH = "./task/tasks.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tasks (
            id TEXT PRIMARY KEY,
            name TEXT,
            status TEXT,
            error TEXT
        )
    """)
    conn.commit()
    conn.close()

def update_status(id, name, status, error):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO tasks (id, name,status, error) 
        VALUES (?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET 
            status=excluded.status,
            error=excluded.error
    """, (id, name, status, error))
    conn.commit()
    conn.close()

def get_task(id):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT name, status, error FROM tasks WHERE id = ?", (id,))
    row = cursor.fetchone()
    conn.close()
    if row:
        name,status,error = row
        return {"id": id, "name":name, "status": status,"error": error}
    else:
        return {"id": id, "name":"Not Found", "status": "Not Found","error": "Not Found"}

def get_all_tasks():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, status, error FROM tasks")
    rows = cursor.fetchall()
    conn.close()
    return [{"id": row[0], "name":row[1], "status": row[2], "error":row[3]} for row in rows]

if __name__=="__main__":
    task=get_task("173177169762787832708")
    print(task)