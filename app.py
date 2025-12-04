import json
import gzip
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import streamlit as st
from openai import OpenAI

# ---------------------------
# Utilidades de entorno
# ---------------------------

def get_openai_client() -> OpenAI | None:
    """
    Inicializa el cliente de OpenAI usando st.secrets.
    """
    api_key = st.secrets.get("OPENAI_API_KEY")
    if not api_key:
        return None
    return OpenAI(api_key=api_key)


# ---------------------------
# Carga de índice vectorial
# ---------------------------

def load_knowledge_vectors(
    json_path: Path,
    gz_path: Path,
    zip_path: Path,
) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    """
    Carga knowledge_vectors desde:
    - JSON plano (si existe),
    - o JSON comprimido (.gz),
    - o ZIP (último recurso).

    Devuelve:
    - docs: lista de dict con texto+metadatos por chunk
    - matrix: matriz numpy con embeddings normalizados (para similitud coseno)
    """

    if json_path.exists():
        # Caso ideal: JSON plano
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    elif gz_path.exists():
        # Caso habitual: JSON comprimido con gzip
        with gzip.open(gz_path, "rt", encoding="utf-8") as f:
            data = json.load(f)
    elif zip_path.exists():
        # Caso alternativo: JSON dentro de un ZIP (primer .json que encontremos)
        with zipfile.ZipFile(zip_path, "r") as zf:
            json_names = [n for n in zf.namelist() if n.endswith(".json")]
            if not json_names:
                raise FileNotFoundError(
                    f"El ZIP {zip_path} no contiene ningún archivo .json"
                )
            with zf.open(json_names[0]) as f:
                data = json.load(f)
    else:
        raise FileNotFoundError(
            f"No se ha encontrado ni {json_path}, ni {gz_path}, ni {zip_path}"
        )

    docs: List[Dict[str, Any]] = []
    vectors: List[List[float]] = []

    for doc in data["documents"]:
        doc_id = doc["doc_id"]
        title = doc.get("title", doc_id)
        source_path = doc.get("source_path")
        for chunk in doc["chunks"]:
            docs.append(
                {
                    "doc_id": doc_id,
                    "title": title,
                    "source_path": source_path,
                    "chunk_id": chunk["chunk_id"],
                    "page": chunk["page"],
                    "position": chunk["position"],
                    "text": chunk["text"],
                }
            )
            vectors.append(chunk["embedding"])

    mat = np.array(vectors, dtype="float32")
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-10
    mat = mat / norms  # normalización para similitud coseno

    return docs, mat


# ---------------------------
# Carga de ontología (YAML)
# ---------------------------

try:
    import yaml  # type: ignore
except ModuleNotFoundError:
    yaml = None


def load_ontology(path: Path) -> Dict[str, Any]:
    """
    Carga la ontología ESG/CSRD/ESRS desde YAML.
    Añade un índice por id de concepto.
    """
    if yaml is None:
        st.warning(
            "La librería 'pyyaml' no está disponible. "
            "No se podrá cargar la ontología."
        )
        return {"concepts": [], "_index": {"by_id": {}}}

    with path.open("r", encoding="utf-8") as f:
        data: Dict[str, Any] = yaml.safe_load(f)

    concepts = data.get("concepts", [])
    by_id = {c["id"]: c for c in concepts}
    data["_index"] = {"by_id": by_id}
    return data


# ---------------------------
# Resolución simple de conceptos
# ---------------------------

def find_matching_concepts(
    ontology: Dict[str, Any],
    question: str,
    max_results: int = 5,
) -> List[Dict[str, Any]]:
    """
    MVP: matching sencillo por palabras clave sobre label/aliases/description.
    """
    q = question.lower()
    concepts = ontology.get("concepts", [])
    matches: List[Dict[str, Any]] = []

    for c in concepts:
        text_parts = [
            c.get("label", ""),
            c.get("description", ""),
        ] + c.get("aliases", [])
        text = " ".join(text_parts).lower()

        # Heurística básica: si alguna palabra de la pregunta aparece en el texto
        if any(token in text for token in q.split()):
            matches.append(c)

    return matches[:max_results]


def build_ontology_context(concepts: List[Dict[str, Any]]) -> str:
    """
    Construye texto legible con los conceptos ontológicos detectados.
    """
    if not concepts:
        return "No se han identificado conceptos ontológicos específicos.\n"

    partes: List[str] = []
    for c in concepts:
        partes.append(f"- {c['id']} ({c.get('type', '')}): {c.get('label')}")
        desc = c.get("description")
        if desc:
            partes.append(f"  Descripción: {desc}")
        refs = c.get("references", [])
        if refs:
            partes.append("  Referencias normativas:")
            for r in refs:
                partes.append(
                    f"    · doc_id={r.get('doc_id')} "
                    f"art={r.get('article') or r.get('section', '')} "
                    f"nota={r.get('note', '')}"
                )
        partes.append("")  # línea en blanco

    return "\n".join(partes)


# ---------------------------
# Búsqueda vectorial y RAG
# ---------------------------

def embed_query(client: OpenAI, query: str, model_name: str = "text-embedding-3-small") -> np.ndarray:
    resp = client.embeddings.create(
        model=model_name,
        input=[query],
    )
    v = np.array(resp.data[0].embedding, dtype="float32")
    v = v / (np.linalg.norm(v) + 1e-10)
    return v


def search_chunks(
    q_vec: np.ndarray,
    docs: List[Dict[str, Any]],
    matrix: np.ndarray,
    k: int = 5,
) -> List[Dict[str, Any]]:
    scores = matrix @ q_vec
    idxs = np.argsort(scores)[::-1][:k]

    hits: List[Dict[str, Any]] = []
    for i in idxs:
        doc = docs[int(i)]
        hits.append(
            {
                **doc,
                "score": float(scores[i]),
            }
        )
    return hits


SYSTEM_PROMPT = """
Eres un asistente experto en normativa de sostenibilidad (CSRD, ESRS, Taxonomía, CSDDD, LIES, IFRS S1/S2).
Respondes apoyándote en:
1) los fragmentos normativos proporcionados (contextos),
2) los conceptos ontológicos y sus relaciones.
Cita siempre la norma (CSRD, ESRS, etc.) y el artículo/sección cuando sea posible.
Si la respuesta no está cubierta por los textos, dilo explícitamente.
"""


def build_context_text(hits: List[Dict[str, Any]]) -> str:
    partes: List[str] = []
    for h in hits:
        header = (
            f"[{h.get('title', h['doc_id'])} | "
            f"pág. {h['page']} | score {h['score']:.3f}]"
        )
        partes.append(header)
        partes.append(h["text"])
        partes.append("")
    return "\n".join(partes)


def answer_with_rag_and_ontology(
    client: OpenAI,
    docs: List[Dict[str, Any]],
    matrix: np.ndarray,
    ontology: Dict[str, Any],
    question: str,
    k_chunks: int = 5,
    k_concepts: int = 5,
    model_name: str = "gpt-4.1-mini",
) -> Dict[str, Any]:
    # 1) Conceptos ontológicos candidatos
    concepts = find_matching_concepts(ontology, question, max_results=k_concepts)
    ontology_text = build_ontology_context(concepts)

    # 2) Embedding de la pregunta
    q_vec = embed_query(client, question)

    # 3) Recuperación de fragmentos normativos
    hits = search_chunks(q_vec, docs, matrix, k=k_chunks)
    context_text = build_context_text(hits)

    user_content = (
        f"PREGUNTA:\n{question}\n\n"
        "CONCEPTOS ONTOLÓGICOS RELACIONADOS:\n"
        f"{ontology_text}\n"
        "FRAGMENTOS NORMATIVOS RELEVANTES:\n"
        f"{context_text}"
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    resp = client.chat.completions.create(
        model=model_name,
        messages=messages,
    )

    answer = resp.choices[0].message.content

    return {
        "answer": answer,
        "hits": hits,
        "concepts": concepts,
    }


# ---------------------------
# Interfaz Streamlit
# ---------------------------

def main() -> None:
    st.set_page_config(
        page_title="GICES · Asistente Normativo (RAG + Ontología)",
        layout="wide",
    )

    st.title("GICES · Asistente Normativo (RAG + Ontología)")
    st.caption("Consulta sobre CSRD, ESRS, Taxonomía, CSDDD, LIES, IFRS S1/S2")

    client = get_openai_client()
    if client is None:
        st.error(
            "No se ha encontrado `OPENAI_API_KEY` en Streamlit Secrets.\n"
            "Configúralo en la consola de Streamlit Cloud."
        )
        return

    base_path = Path(__file__).parent
    kv_json_path = base_path / "rag" / "knowledge_vec
