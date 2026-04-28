"""
agent_nodes.py — Nós do agente LangGraph para o assistente de e-commerce ShopFácil.

Fluxo:
  1. nó_entrada        → prepara o estado inicial
  2. nó_classificacao  → identifica a intenção do usuário (via LLM)
  3. nó_recuperacao    → busca contexto na base vetorial (RAG)
  4. nó_decisao        → decide o tipo de resposta adequada
  5. nó_geracao        → monta o prompt e gera a resposta final (via LLM)
"""

import os
from groq import Groq
from agent_state import AgentState
from vectorstore import search

# ─── Cliente Groq ────────────────────────────────────────────────────────────
_groq = Groq(api_key=os.environ.get("GROQ_API_KEY"))
MODEL = "llama3-8b-8192"


def _call_llm(prompt: str, max_tokens: int = 600) -> str:
    """Chamada simples ao Groq."""
    response = _groq.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()


# ─── Nó 1: Entrada ───────────────────────────────────────────────────────────
def no_entrada(state: AgentState) -> AgentState:
    """
    Recebe a pergunta do usuário e inicializa o estado do agente.
    Não realiza nenhum processamento — apenas garante que os campos
    obrigatórios estejam presentes.
    """
    return {
        **state,
        "intent": "",
        "context": "",
        "has_context": False,
        "response_type": "",
        "final_response": "",
    }


# ─── Nó 2: Classificação de intenção ─────────────────────────────────────────
def no_classificacao(state: AgentState) -> AgentState:
    """
    Usa a LLM para classificar a intenção da pergunta do usuário.
    Categorias possíveis:
      - troca_devolucao
      - pagamento
      - entrega
      - cancelamento
      - produto
      - outro
    """
    prompt = f"""Você é um classificador de intenções para um e-commerce.
Classifique a mensagem do cliente em APENAS UMA das categorias abaixo:

- troca_devolucao : perguntas sobre troca de produto, devolução, arrependimento, reembolso
- pagamento       : perguntas sobre formas de pagamento, parcelamento, PIX, boleto, cartão, desconto
- entrega         : perguntas sobre prazo de entrega, rastreamento, frete, transportadora, status do pedido
- cancelamento    : perguntas sobre como cancelar um pedido
- produto         : perguntas sobre disponibilidade, estoque, garantia, especificações do produto
- outro           : qualquer outra pergunta não listada acima

Mensagem do cliente: "{state['question']}"

Responda SOMENTE com o nome da categoria, sem explicações, sem pontuação."""

    raw = _call_llm(prompt, max_tokens=20).lower().strip()

    valid = {"troca_devolucao", "pagamento", "entrega", "cancelamento", "produto", "outro"}
    intent = raw if raw in valid else "outro"

    print(f"[Classificação] Intenção identificada: {intent}")
    return {**state, "intent": intent}


# ─── Nó 3: Recuperação de contexto (RAG) ─────────────────────────────────────
def no_recuperacao(state: AgentState) -> AgentState:
    """
    Consulta a base vetorial com a pergunta do usuário e recupera
    os trechos mais relevantes da base de conhecimento.
    """
    docs = search(state["question"], k=4)

    if docs:
        context = "\n\n---\n\n".join(docs)
        has_context = True
        print(f"[RAG] {len(docs)} chunk(s) recuperado(s).")
    else:
        context = ""
        has_context = False
        print("[RAG] Nenhum chunk relevante encontrado.")

    return {**state, "context": context, "has_context": has_context}


# ─── Nó 4: Decisão ───────────────────────────────────────────────────────────
def no_decisao(state: AgentState) -> AgentState:
    """
    Decide o tipo de resposta a ser gerada com base na intenção
    classificada e nas palavras-chave presentes na pergunta.

    Tipos de resposta:
      - checklist      : passo a passo numerado
      - politica       : explicação de política/regra
      - resumo         : resumo objetivo
      - resposta_direta: resposta direta e objetiva
      - sem_informacao : não há contexto suficiente
    """
    if not state["has_context"]:
        response_type = "sem_informacao"
    else:
        q = state["question"].lower()
        intent = state["intent"]

        if any(kw in q for kw in ["como faço", "passo a passo", "processo", "etapas", "passos"]):
            response_type = "checklist"
        elif any(kw in q for kw in ["resumo", "resumir", "resumindo", "em poucas palavras"]):
            response_type = "resumo"
        elif intent in ("troca_devolucao", "cancelamento"):
            response_type = "politica"
        else:
            response_type = "resposta_direta"

    print(f"[Decisão] Tipo de resposta: {response_type}")
    return {**state, "response_type": response_type}


# ─── Nó 5: Geração da resposta final ─────────────────────────────────────────
def no_geracao(state: AgentState) -> AgentState:
    """
    Monta o prompt final e chama a LLM (Groq) para gerar a resposta.
    O prompt é adaptado conforme o tipo de resposta decidido no nó anterior.
    """

    # Instruções de formato por tipo de resposta
    format_map = {
        "checklist": (
            "Responda em formato de lista numerada com passos claros e objetivos. "
            "Cada passo deve ser uma ação concreta."
        ),
        "politica": (
            "Explique a política de forma clara. Destaque prazos, condições e "
            "exceções importantes. Seja direto e informativo."
        ),
        "resumo": (
            "Forneça um resumo conciso em no máximo 3 parágrafos curtos, "
            "cobrindo os pontos mais importantes."
        ),
        "resposta_direta": (
            "Responda de forma direta e objetiva, sem rodeios."
        ),
        "sem_informacao": (
            "Informe ao cliente que não encontrou informações específicas sobre "
            "o assunto e oriente-o a entrar em contato com o suporte."
        ),
    }

    format_instruction = format_map.get(state["response_type"], format_map["resposta_direta"])

    if state["has_context"]:
        context_section = f"""Base de conhecimento consultada:
{state['context']}"""
    else:
        context_section = "Nenhuma informação específica foi encontrada na base de conhecimento."

    prompt = f"""Você é o assistente virtual da ShopFácil, uma loja online brasileira.
Seu tom é cordial, profissional e prestativo. Responda sempre em português brasileiro.

{format_instruction}

{context_section}

Pergunta do cliente: {state['question']}

Caso o contexto não cubra completamente a pergunta, seja honesto e indique o canal de suporte
(chat online: seg-sab 8h-20h | telefone 0800 123 4567 | e-mail: atendimento@shopfacil.com.br).

Resposta:"""

    answer = _call_llm(prompt, max_tokens=700)
    print(f"[Geração] Resposta gerada ({len(answer)} caracteres).")
    return {**state, "final_response": answer}
