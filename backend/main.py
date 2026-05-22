import asyncio
import os
from fastapi import (
    FastAPI,
    UploadFile,
    WebSocket,
    WebSocketDisconnect
)
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from agents.orchestrator import AgentOrchestrator
from agents.file_agent import FileAgent  

load_dotenv()

app = FastAPI(title="AgentML API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

os.makedirs("data",   exist_ok=True)
os.makedirs("models", exist_ok=True)

connections = []

async def broadcast(msg: str):
    for ws in connections:
        try:
            await ws.send_text(msg)
        except Exception:
            pass

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    connections.append(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        connections.remove(ws)

@app.post("/api/run")
async def run_pipeline(file: UploadFile):
    path = f"data/{file.filename}"
    with open(path, "wb") as f:
        f.write(await file.read())
    agent = AgentOrchestrator(
        broadcast=broadcast
    )
    asyncio.create_task(agent.run(path))
    return {
        "status": "started",
        "file":   file.filename
    }

@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "agent":  "LangChain + TinyLlama FREE"
    }

@app.post("/api/analyse-file")
async def analyse_file(
    file: UploadFile,
    question: str = "Analyse this file in detail"
):
    path = f"data/{file.filename}"
    with open(path, "wb") as f:
        f.write(await file.read())
    agent  = FileAgent()
    result = agent.analyse(path, question)
    return result

@app.get("/api/supported-types")
def supported_types():
    return {
        "supported_file_types": [
            "CSV (.csv) → ML Pipeline",
            "Excel (.xlsx .xls) → Data Analysis",
            "Word (.docx) → Text Extraction",
            "PDF (.pdf) → Document Reading",
            "Images (.jpg .png .gif) → AI Vision",
            "Text (.txt .md) → Summarization"
        ]
    }
