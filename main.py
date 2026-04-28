"""
main.py — Backend FastAPI do Assistente ShopFácil.

Endpoints:
  GET  /          → serve o index.html da interface web
  POST /perguntar → recebe a pergunta e executa o agente LangGraph
  GET  /status    → verifica se a base vetorial está carregada
"""

import os
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

# Verifica chave da API antes de iniciar
if not os.environ.get("GROQ_API_KEY"):
    raise RuntimeError(
        "GROQ_API_KEY não definida. Crie um arquivo .env com GROQ_API_KEY=sua_chave"
    )

from agent_graph import agent
from agent_state import AgentState
from vectorstore import collection_count

app = FastAPI(
    title="Assistente ShopFácil",
    description="Agente de atendimento ao cliente com LangGraph + RAG + Groq",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Modelos de dados ──────────────────────────────────────────────────────────

class PerguntaRequest(BaseModel):
    pergunta: str


class RespostaResponse(BaseModel):
    resposta: str
    intencao: str
    tipo_resposta: str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    """Serve a interface web."""
    html_path = Path(__file__).parent / "frontend" / "index.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="index.html não encontrado.")
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.get("/status")
def status():
    """Verifica o estado da base vetorial."""
    count = collection_count()
    return {
        "status": "ok" if count > 0 else "base_vazia",
        "chunks_na_base": count,
        "mensagem": (
            f"{count} chunks disponíveis na base de conhecimento."
            if count > 0
            else "Base vazia. Execute: python ingest.py"
        ),
    }


@app.post("/perguntar", response_model=RespostaResponse)
def perguntar(body: PerguntaRequest):
    """
    Recebe a pergunta do usuário, executa o agente LangGraph e retorna
    a resposta gerada, a intenção classificada e o tipo de resposta.
    """
    pergunta = body.pergunta.strip()

    if not pergunta:
        raise HTTPException(status_code=400, detail="A pergunta não pode ser vazia.")

    if len(pergunta) > 1000:
        raise HTTPException(status_code=400, detail="Pergunta muito longa (máx. 1000 caracteres).")

    # Estado inicial do agente
    estado_inicial: AgentState = {
        "question": pergunta,
        "intent": "",
        "context": "",
        "has_context": False,
        "response_type": "",
        "final_response": "",
    }

    try:
        resultado = agent.invoke(estado_inicial)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro no agente: {str(e)}")

    return RespostaResponse(
        resposta=resultado["final_response"],
        intencao=resultado["intent"],
        tipo_resposta=resultado["response_type"],
    )