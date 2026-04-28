"""
agent_graph.py — Definição e compilação do grafo LangGraph.

Estrutura do fluxo:

  START
    │
    ▼
  no_entrada          ← recebe a pergunta do usuário
    │
    ▼
  no_classificacao    ← classifica a intenção (LLM)
    │
    ▼
  no_recuperacao      ← busca contexto na base vetorial (RAG)
    │
    ▼
  no_decisao          ← decide o tipo de resposta
    │                    (condicional: com ou sem contexto)
    ▼
  no_geracao          ← gera a resposta final (LLM)
    │
    ▼
   END
"""

from langgraph.graph import StateGraph, END, START
from agent_state import AgentState
from agent_nodes import (
    no_entrada,
    no_classificacao,
    no_recuperacao,
    no_decisao,
    no_geracao,
)


def _rota_apos_recuperacao(state: AgentState) -> str:
    """
    Nó de decisão condicional:
    - Se há contexto recuperado → vai para 'decisao' (escolhe o formato)
    - Se NÃO há contexto         → pula direto para 'geracao' (sem_informacao)
    """
    if state.get("has_context"):
        return "decisao"
    return "geracao"


def build_graph():
    """Constrói e compila o grafo do agente."""

    graph = StateGraph(AgentState)

    # ── Registra os nós ──────────────────────────────────────────────────────
    graph.add_node("entrada",       no_entrada)
    graph.add_node("classificacao", no_classificacao)
    graph.add_node("recuperacao",   no_recuperacao)
    graph.add_node("decisao",       no_decisao)
    graph.add_node("geracao",       no_geracao)

    # ── Define as arestas ────────────────────────────────────────────────────
    graph.set_entry_point("entrada")
    graph.add_edge("entrada",       "classificacao")
    graph.add_edge("classificacao", "recuperacao")

    # Aresta condicional: verifica se há contexto antes de decidir o formato
    graph.add_conditional_edges(
        "recuperacao",
        _rota_apos_recuperacao,
        {
            "decisao": "decisao",   # há contexto → nó de decisão de formato
            "geracao": "geracao",   # sem contexto → gera resposta diretamente
        },
    )

    graph.add_edge("decisao",  "geracao")
    graph.add_edge("geracao",  END)

    return graph.compile()


# Instância pronta para ser importada
agent = build_graph()