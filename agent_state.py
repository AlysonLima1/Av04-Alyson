from typing import TypedDict, Optional


class AgentState(TypedDict):
    """Estado compartilhado entre os nós do agente LangGraph."""
    question: str          # Pergunta original do usuário
    intent: str            # Intenção classificada
    context: str           # Contexto recuperado via RAG
    has_context: bool      # Se há contexto suficiente
    response_type: str     # Tipo de resposta a gerar
    final_response: str    # Resposta final gerada pela LLM
