# Av04-Alyson# 🛍️ ShopFácil — Assistente Virtual de Atendimento ao Cliente

Projeto da Atividade Avaliativa 04 — Tópicos Avançados de IA  
Curso: Sistemas para Internet

---

## 📋 Descrição do Tema

Assistente virtual de atendimento ao cliente para um e-commerce fictício chamado **ShopFácil**.  
O agente responde perguntas sobre trocas, devoluções, formas de pagamento, entregas, cancelamentos e FAQ geral.

---

## 🧠 Descrição da Base de Conhecimento

A base é composta por 5 arquivos `.txt` na pasta `knowledge_base/`:

| Arquivo | Conteúdo |
|---|---|
| `01_politica_troca_devolucao.txt` | Política de trocas, devoluções e reembolsos |
| `02_formas_pagamento.txt` | Formas de pagamento, parcelamento, PIX, cartão |
| `03_entrega_rastreamento.txt` | Prazos de entrega, modalidades de frete e rastreamento |
| `04_cancelamentos.txt` | Como cancelar pedidos e prazos de reembolso |
| `05_faq_geral.txt` | FAQ geral: cadastro, produtos, cupons, segurança |

Os arquivos são divididos em chunks e armazenados no **ChromaDB** (banco vetorial local).  
A busca usa embeddings semânticos (`all-MiniLM-L6-v2`) para recuperar trechos relevantes.

---

## 🔄 Fluxo do Agente (LangGraph)

```
START
  │
  ▼
no_entrada          ← recebe a pergunta do usuário e inicializa o estado
  │
  ▼
no_classificacao    ← classifica a INTENÇÃO via LLM (Groq)
  │                   categorias: troca_devolucao | pagamento | entrega |
  │                               cancelamento | produto | outro
  ▼
no_recuperacao      ← busca os chunks mais relevantes na base vetorial (RAG)
  │
  ▼ (aresta condicional)
  ├── se há contexto → no_decisao ← decide o TIPO de resposta:
  │                                 checklist | politica | resumo | resposta_direta
  │
  └── se sem contexto → no_geracao (direto, resposta "sem_informacao")
  │
  ▼
no_geracao          ← monta o prompt final e chama a LLM (Groq / LLaMA 3)
  │
  ▼
END → retorna resposta ao usuário
```

### O que caracteriza o comportamento de agente:
1. **Classificação de intenção** via LLM — não é hardcoded
2. **Aresta condicional** no grafo — dois caminhos possíveis após a recuperação
3. **Decisão de formato** de resposta baseada em intenção + palavras-chave da pergunta
4. **Adaptação do prompt** conforme o tipo de resposta decidido

---

## 🗂️ Estrutura do Projeto

```
ecommerce-agent/
├── README.md
├── requirements.txt
├── .env.example
│
├── knowledge_base/          ← base de conhecimento (TXT)
│   ├── 01_politica_troca_devolucao.txt
│   ├── 02_formas_pagamento.txt
│   ├── 03_entrega_rastreamento.txt
│   ├── 04_cancelamentos.txt
│   └── 05_faq_geral.txt
│
├── agent_state.py           ← TypedDict com o estado compartilhado
├── agent_nodes.py           ← todos os nós do agente LangGraph
├── agent_graph.py           ← definição e compilação do grafo
├── vectorstore.py           ← wrapper do ChromaDB (busca vetorial)
├── ingest.py                ← script de ingestão da base de conhecimento
├── main.py                  ← servidor FastAPI (backend)
│
└── frontend/
    └── index.html           ← interface web
```

---

## 🚀 Instruções de Execução

### 1. Pré-requisitos
- Python 3.11+
- Conta gratuita no Groq: https://console.groq.com

### 2. Clone e instale as dependências

```bash
git clone <url-do-repositorio>
cd ecommerce-agent

pip install -r requirements.txt
```

### 3. Configure a chave de API

```bash
cp .env.example .env
# Abra o .env e substitua "sua_chave_aqui" pela sua chave do Groq
```

### 4. Ingira a base de conhecimento (execute apenas uma vez)

```bash
python ingest.py
```

Saída esperada:
```
📄 Processando: 01_politica_troca_devolucao.txt → 12 chunks
📄 Processando: 02_formas_pagamento.txt → 10 chunks
...
✅ Ingestão concluída! 52 chunks disponíveis na base vetorial.
```

### 5. Inicie o servidor

```bash
uvicorn main:app --reload
```

### 6. Acesse a interface

Abra o navegador em: **http://localhost:8000**

---

## 🛠️ Tecnologias Utilizadas

| Tecnologia | Uso |
|---|---|
| **LangGraph** | Orquestração do fluxo do agente |
| **Groq + LLaMA 3** | LLM para classificação e geração de respostas |
| **ChromaDB** | Banco vetorial local para RAG |
| **FastAPI** | Backend/API REST |
| **Python** | Linguagem principal |
| **HTML/CSS/JS** | Interface web |

---

## 📡 Endpoints da API

| Método | Rota | Descrição |
|---|---|---|
| `GET` | `/` | Serve a interface web |
| `POST` | `/perguntar` | Processa a pergunta via agente |
| `GET` | `/status` | Verifica o estado da base vetorial |

### Exemplo de chamada à API:
```bash
curl -X POST http://localhost:8000/perguntar \
  -H "Content-Type: application/json" \
  -d '{"pergunta": "Como faço para trocar um produto?"}'
```

### Resposta:
```json
{
  "resposta": "Para solicitar a troca de um produto...",
  "intencao": "troca_devolucao",
  "tipo_resposta": "politica"
}
```

---

## 👥 Equipe

- [Nome dos integrantes]

---

## 📎 Dependências

```
fastapi==0.115.6
uvicorn==0.32.1
python-dotenv==1.0.1
groq==0.12.0
langgraph==0.2.53
chromadb==0.5.23
pydantic==2.10.3
```
