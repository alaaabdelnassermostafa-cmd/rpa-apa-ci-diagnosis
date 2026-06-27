import os
import json
import sqlite3
import threading
from typing import Dict, Any
from pathlib import Path
from fastapi import FastAPI, Request, BackgroundTasks, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# Adjust import paths for the APA module
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.apa.agent import run_agent

app = FastAPI(title="APA Webhook & Dashboard Server")

# Mount static files (CSS, JS)
STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

DB_PATH = Path("data/dashboard.sqlite3")

def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS failures (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            repo TEXT,
            branch TEXT,
            commit_sha TEXT,
            status TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            classification_category TEXT,
            remediation_plan TEXT,
            raw_event TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

def process_github_webhook(payload: Dict[str, Any]):
    """Background task to run the APA pipeline when a webhook arrives."""
    repo = payload.get("repository", {}).get("full_name", "unknown/repo")
    workflow_run = payload.get("workflow_run", {})
    branch = workflow_run.get("head_branch", "unknown")
    commit_sha = workflow_run.get("head_sha", "unknown")
    
    # Store initial pending state in DB
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO failures (repo, branch, commit_sha, status, raw_event) VALUES (?, ?, ?, ?, ?)",
        (repo, branch, commit_sha, "PROCESSING", json.dumps(payload))
    )
    failure_id = cursor.lastrowid
    conn.commit()
    conn.close()

    try:
        # Run the APA Pipeline using the raw webhook payload (or a synthesized structure)
        # In a real system, you'd fetch the exact jobs and logs via GitHub API.
        # For the dashboard demo, we just pass the webhook payload as raw_run.
        result = run_agent(payload)
        
        category = result.get("classification", {}).get("category", "UNKNOWN")
        remediation_markdown = result.get("classification", {}).get("remediation_markdown", "No remediation generated.")
        
        # Update DB with results
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE failures SET status = ?, classification_category = ?, remediation_plan = ? WHERE id = ?",
            ("COMPLETED", category, remediation_markdown, failure_id)
        )
        conn.commit()
        conn.close()
        
    except Exception as e:
        # Handle failure
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE failures SET status = ?, classification_category = ?, remediation_plan = ? WHERE id = ?",
            ("ERROR", "SYSTEM_ERROR", f"Failed to process webhook: {str(e)}", failure_id)
        )
        conn.commit()
        conn.close()

@app.post("/webhook")
async def github_webhook(request: Request, background_tasks: BackgroundTasks):
    """GitHub Webhook Endpoint. Triggers the pipeline on workflow_run failure."""
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON payload")

    # Only process completed, failed workflow runs
    if payload.get("action") == "completed" and "workflow_run" in payload:
        if payload["workflow_run"].get("conclusion") == "failure":
            background_tasks.add_task(process_github_webhook, payload)
            return {"status": "accepted", "message": "Processing failure in background."}
            
    return {"status": "ignored", "message": "Not a workflow_run failure event."}

@app.get("/api/failures")
def get_failures():
    """API Endpoint to fetch recent failures for the UI Dashboard."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM failures ORDER BY timestamp DESC LIMIT 20")
    rows = cursor.fetchall()
    conn.close()
    
    return [dict(row) for row in rows]

@app.get("/", response_class=HTMLResponse)
def serve_dashboard():
    """Serve the main UI Dashboard HTML."""
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return index_path.read_text(encoding="utf-8")
    return "<h1>Dashboard UI not built yet</h1>"

if __name__ == "__main__":
    import uvicorn
    # Make sure we use the correct LLM defaults for the server
    os.environ.setdefault("CI_AGENT_MODEL", "deepseek-chat")
    os.environ.setdefault("LLM_PROVIDER", "deepseek")
    
    port = int(os.environ.get("WEBHOOK_PORT", 8090))
    print(f"Starting APA Webhook & Dashboard Server on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)
