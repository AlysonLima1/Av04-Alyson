import os
import chromadb
from chromadb.utils import embedding_functions

# Diretório onde o ChromaDB salva os dados
CHROMA_DB_DIR = "./chroma_db"
COLLECTION_NAME = "shopfacil_kb"

# Função de embedding padrão (sentence-transformers all-MiniLM-L6-v2)
_ef = embedding_functions.DefaultEmbeddingFunction()

_client = None
_collection = None


def _get_collection():
    """Retorna (ou cria) a coleção ChromaDB."""
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
        _collection = _client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=_ef,
        )
    return _collection


def add_documents(docs: list[dict]):
    """
    Adiciona documentos à base vetorial.
    Cada doc deve ter: {'id': str, 'text': str, 'metadata': dict}
    """
    col = _get_collection()
    col.add(
        ids=[d["id"] for d in docs],
        documents=[d["text"] for d in docs],
        metadatas=[d["metadata"] for d in docs],
    )
    print(f"[VectorStore] {len(docs)} chunks adicionados à coleção '{COLLECTION_NAME}'.")


def search(query: str, k: int = 4) -> list[str]:
    """
    Busca os k chunks mais relevantes para a query.
    Retorna lista de strings com o conteúdo dos documentos.
    """
    col = _get_collection()
    results = col.query(
        query_texts=[query],
        n_results=min(k, col.count()),
    )
    if not results or not results["documents"]:
        return []
    return results["documents"][0]  # lista de strings


def collection_count() -> int:
    """Retorna o número de chunks na coleção."""
    return _get_collection().count()