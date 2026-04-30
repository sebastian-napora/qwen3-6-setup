#!/usr/bin/env python3
"""Local embedding and RAG endpoints for the Qwen serving stack."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import sqlite3
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from fastapi import APIRouter, HTTPException, Request

logger = logging.getLogger("qwen_local_rag")

BASE_DIR = Path(__file__).parent.resolve()
RAG_DIR = Path(os.environ.get("QWEN_RAG_DIR", BASE_DIR / "rag_data")).resolve()
DB_PATH = Path(os.environ.get("QWEN_RAG_DB", RAG_DIR / "rag.sqlite3")).resolve()
DOCUMENT_ROOT = Path(
    os.environ.get("QWEN_RAG_DOCUMENT_ROOT", BASE_DIR / "documents")
).resolve()

DEFAULT_EMBED_MODEL = os.environ.get(
    "QWEN_RAG_EMBED_MODEL", "unsloth/Qwen3-Embedding-4B"
)
DEFAULT_EMBED_BACKEND = os.environ.get("QWEN_RAG_EMBED_BACKEND", "auto")
DEFAULT_CHAT_MODEL = os.environ.get("QWEN_RAG_CHAT_MODEL", "qwen3.6-35b-nvfp4")
DEFAULT_QUERY_INSTRUCTION = os.environ.get(
    "QWEN_RAG_QUERY_INSTRUCTION",
    "Given a user question, retrieve relevant passages from local documents that answer it.",
)
DEFAULT_CHUNK_SIZE = int(os.environ.get("QWEN_RAG_CHUNK_SIZE", "1400"))
DEFAULT_CHUNK_OVERLAP = int(os.environ.get("QWEN_RAG_CHUNK_OVERLAP", "180"))
DEFAULT_TOP_K = int(os.environ.get("QWEN_RAG_TOP_K", "6"))
DEFAULT_MAX_CONTEXT_CHARS = int(os.environ.get("QWEN_RAG_MAX_CONTEXT_CHARS", "18000"))
DEFAULT_EMBED_MAX_LENGTH = int(os.environ.get("QWEN_RAG_EMBED_MAX_LENGTH", "8192"))
DEFAULT_EMBED_BATCH_SIZE = int(os.environ.get("QWEN_RAG_EMBED_BATCH_SIZE", "8"))
MAX_FILE_BYTES = int(os.environ.get("QWEN_RAG_MAX_FILE_BYTES", str(25 * 1024 * 1024)))

EMBED_MODEL_PRESETS = {
    "mlx-8b": "mlx-community/Qwen3-Embedding-8B-4bit-DWQ",
    "mlx-qwen3-8b-4bit": "mlx-community/Qwen3-Embedding-8B-4bit-DWQ",
    "unsloth-4b": "unsloth/Qwen3-Embedding-4B",
    "qwen-4b": "Qwen/Qwen3-Embedding-4B",
    "qwen-0.6b": "Qwen/Qwen3-Embedding-0.6B",
}

TEXT_SUFFIXES = {
    ".c",
    ".cc",
    ".cpp",
    ".css",
    ".csv",
    ".go",
    ".h",
    ".html",
    ".java",
    ".js",
    ".json",
    ".jsx",
    ".log",
    ".md",
    ".mdx",
    ".php",
    ".py",
    ".rb",
    ".rs",
    ".sh",
    ".sql",
    ".swift",
    ".toml",
    ".ts",
    ".tsx",
    ".txt",
    ".xml",
    ".yaml",
    ".yml",
}
SUPPORTED_SUFFIXES = TEXT_SUFFIXES | {".pdf"}

router = APIRouter(tags=["local-rag"])


@dataclass
class SourceDocument:
    source: str
    text: str
    metadata: Dict[str, Any]


@dataclass
class ChunkRecord:
    collection: str
    source: str
    chunk_index: int
    text: str
    metadata: Dict[str, Any]


def resolve_embedding_model(model_id: Optional[str]) -> str:
    if not model_id:
        model_id = DEFAULT_EMBED_MODEL
    return EMBED_MODEL_PRESETS.get(str(model_id).strip().lower(), str(model_id).strip())


def resolve_embedding_backend(model_id: str, backend: Optional[str] = None) -> str:
    requested = str(backend or DEFAULT_EMBED_BACKEND or "auto").strip().lower()
    if requested in {"", "auto"}:
        return "mlx" if model_id.startswith("mlx-community/") else "hf"
    if requested in {"hf", "huggingface", "transformers", "torch"}:
        return "hf"
    if requested == "mlx":
        return "mlx"
    raise ValueError(
        "embedding_backend must be one of: auto, mlx, hf, huggingface, transformers"
    )


class MLXEmbeddingBackend:
    """Lazy singleton for local Qwen3 embedding models."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._model: Any = None
        self._processor: Any = None
        self._generate: Any = None
        self._model_id: Optional[str] = None
        self._loaded_at: Optional[float] = None

    @property
    def loaded_model(self) -> Optional[str]:
        return self._model_id

    def _ensure_loaded(self, model_id: str) -> None:
        with self._lock:
            if self._model is not None and self._model_id == model_id:
                return

            try:
                from mlx_embeddings import generate, load
            except ModuleNotFoundError as exc:
                raise RuntimeError(
                    "mlx-embeddings is required for local Qwen embeddings. "
                    "Install it with: pip install mlx-embeddings"
                ) from exc
            except Exception as exc:
                raise RuntimeError(
                    "Failed to initialize MLX embeddings. On macOS this usually "
                    "means the current process cannot access a Metal device; run "
                    "the proxy from a normal terminal session."
                ) from exc

            logger.info("Loading local embedding model: %s", model_id)
            try:
                model, processor = load(model_id)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to load embedding model {model_id!r}. If this is the "
                    "first run, ensure Hugging Face access is available so the "
                    "model can be downloaded."
                ) from exc

            self._model = model
            self._processor = processor
            self._generate = generate
            self._model_id = model_id
            self._loaded_at = time.time()
            logger.info("Local embedding model loaded: %s", model_id)

    def embed_texts(
        self,
        texts: Sequence[str],
        model_id: str = DEFAULT_EMBED_MODEL,
        max_length: int = DEFAULT_EMBED_MAX_LENGTH,
        batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
    ) -> List[np.ndarray]:
        if not texts:
            return []

        self._ensure_loaded(model_id)
        assert self._model is not None
        assert self._processor is not None
        assert self._generate is not None

        vectors: List[np.ndarray] = []
        with self._lock:
            for start in range(0, len(texts), batch_size):
                batch = list(texts[start : start + batch_size])
                output = self._generate(
                    self._model,
                    self._processor,
                    texts=batch,
                    max_length=max_length,
                    padding=True,
                    truncation=True,
                )
                embeds = getattr(output, "text_embeds", None)
                if embeds is None:
                    raise RuntimeError("Embedding model did not return text_embeds")

                arr = np.asarray(embeds.tolist(), dtype=np.float32)
                norms = np.linalg.norm(arr, axis=1, keepdims=True)
                arr = arr / np.maximum(norms, 1e-12)
                vectors.extend(arr[i].astype(np.float32) for i in range(arr.shape[0]))

        return vectors

    async def aembed_texts(
        self,
        texts: Sequence[str],
        model_id: str = DEFAULT_EMBED_MODEL,
        max_length: int = DEFAULT_EMBED_MAX_LENGTH,
        batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
    ) -> List[np.ndarray]:
        return await asyncio.to_thread(
            self.embed_texts,
            texts,
            model_id,
            max_length,
            batch_size,
        )


class HFEmbeddingBackend:
    """Lazy Torch/Transformers backend for Safetensors Qwen3 embedding models."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._model: Any = None
        self._tokenizer: Any = None
        self._torch: Any = None
        self._model_id: Optional[str] = None
        self._device: Optional[str] = None
        self._loaded_at: Optional[float] = None

    @property
    def loaded_model(self) -> Optional[str]:
        return self._model_id

    @property
    def device(self) -> Optional[str]:
        return self._device

    def _ensure_loaded(self, model_id: str) -> None:
        with self._lock:
            if self._model is not None and self._model_id == model_id:
                return

            try:
                import torch
                from transformers import AutoModel, AutoTokenizer
            except ModuleNotFoundError as exc:
                raise RuntimeError(
                    "The Hugging Face embedding backend requires torch and transformers. "
                    "Install them with: ./install.sh --hf-embeddings"
                ) from exc

            device = os.environ.get("QWEN_RAG_HF_DEVICE")
            if not device:
                if torch.cuda.is_available():
                    device = "cuda"
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    device = "mps"
                else:
                    device = "cpu"

            torch_dtype = self._resolve_dtype(torch)
            model_kwargs: Dict[str, Any] = {"torch_dtype": torch_dtype}
            attn_implementation = os.environ.get("QWEN_RAG_HF_ATTN_IMPLEMENTATION")
            if attn_implementation:
                model_kwargs["attn_implementation"] = attn_implementation
            if os.environ.get("QWEN_RAG_HF_TRUST_REMOTE_CODE", "0") == "1":
                model_kwargs["trust_remote_code"] = True

            logger.info("Loading HF embedding model: %s on %s", model_id, device)
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
                model = AutoModel.from_pretrained(model_id, **model_kwargs)
                model.to(device)
                model.eval()
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to load Hugging Face embedding model {model_id!r}. "
                    "If this is the first run, ensure Hugging Face access is available "
                    "or pre-download it with ./download_embedding_model.sh."
                ) from exc

            self._torch = torch
            self._tokenizer = tokenizer
            self._model = model
            self._model_id = model_id
            self._device = device
            self._loaded_at = time.time()
            logger.info("HF embedding model loaded: %s on %s", model_id, device)

    def _resolve_dtype(self, torch: Any) -> Any:
        requested = os.environ.get("QWEN_RAG_HF_DTYPE", "auto").strip().lower()
        if requested == "auto":
            if torch.cuda.is_available():
                return torch.bfloat16
            return torch.float32
        if requested in {"bf16", "bfloat16"}:
            return torch.bfloat16
        if requested in {"fp16", "float16", "half"}:
            return torch.float16
        if requested in {"fp32", "float32"}:
            return torch.float32
        raise RuntimeError("QWEN_RAG_HF_DTYPE must be one of: auto, bfloat16, float16, float32")

    def embed_texts(
        self,
        texts: Sequence[str],
        model_id: str,
        max_length: int = DEFAULT_EMBED_MAX_LENGTH,
        batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
    ) -> List[np.ndarray]:
        if not texts:
            return []

        self._ensure_loaded(model_id)
        assert self._torch is not None
        assert self._tokenizer is not None
        assert self._model is not None
        assert self._device is not None

        torch = self._torch
        vectors: List[np.ndarray] = []
        with self._lock:
            with torch.inference_mode():
                for start in range(0, len(texts), batch_size):
                    batch = list(texts[start : start + batch_size])
                    batch_dict = self._tokenizer(
                        batch,
                        padding=True,
                        truncation=True,
                        max_length=max_length,
                        return_tensors="pt",
                    )
                    batch_dict = {
                        key: value.to(self._device) for key, value in batch_dict.items()
                    }
                    outputs = self._model(**batch_dict)
                    embeddings = _last_token_pool(
                        outputs.last_hidden_state,
                        batch_dict["attention_mask"],
                        torch,
                    )
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                    arr = embeddings.detach().cpu().float().numpy().astype(np.float32)
                    vectors.extend(arr[i] for i in range(arr.shape[0]))

        return vectors

    async def aembed_texts(
        self,
        texts: Sequence[str],
        model_id: str,
        max_length: int = DEFAULT_EMBED_MAX_LENGTH,
        batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
    ) -> List[np.ndarray]:
        return await asyncio.to_thread(
            self.embed_texts,
            texts,
            model_id,
            max_length,
            batch_size,
        )


class LocalEmbeddingService:
    """Selects the right local embedding backend for each model."""

    def __init__(self) -> None:
        self._mlx = MLXEmbeddingBackend()
        self._hf = HFEmbeddingBackend()
        self._last_backend: Optional[str] = None
        self._last_model: Optional[str] = None

    @property
    def loaded_model(self) -> Optional[str]:
        if self._last_backend and self._last_model:
            return f"{self._last_backend}:{self._last_model}"
        return None

    @property
    def loaded_models(self) -> Dict[str, Optional[str]]:
        return {
            "mlx": self._mlx.loaded_model,
            "hf": self._hf.loaded_model,
        }

    def backend_status(self) -> Dict[str, Any]:
        return {
            "default_backend": DEFAULT_EMBED_BACKEND,
            "loaded_models": self.loaded_models,
            "hf_device": self._hf.device,
        }

    def embed_texts(
        self,
        texts: Sequence[str],
        model_id: str = DEFAULT_EMBED_MODEL,
        backend: Optional[str] = None,
        max_length: int = DEFAULT_EMBED_MAX_LENGTH,
        batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
    ) -> List[np.ndarray]:
        resolved_model = resolve_embedding_model(model_id)
        resolved_backend = resolve_embedding_backend(resolved_model, backend)
        if resolved_backend == "mlx":
            vectors = self._mlx.embed_texts(
                texts,
                model_id=resolved_model,
                max_length=max_length,
                batch_size=batch_size,
            )
        else:
            vectors = self._hf.embed_texts(
                texts,
                model_id=resolved_model,
                max_length=max_length,
                batch_size=batch_size,
            )
        self._last_backend = resolved_backend
        self._last_model = resolved_model
        return vectors

    async def aembed_texts(
        self,
        texts: Sequence[str],
        model_id: str = DEFAULT_EMBED_MODEL,
        backend: Optional[str] = None,
        max_length: int = DEFAULT_EMBED_MAX_LENGTH,
        batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
    ) -> List[np.ndarray]:
        return await asyncio.to_thread(
            self.embed_texts,
            texts,
            model_id,
            backend,
            max_length,
            batch_size,
        )


class LocalVectorStore:
    """SQLite metadata store with NumPy cosine search over normalized vectors."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._lock = Lock()
        self._initialized = False

    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init(self) -> None:
        if self._initialized:
            return
        with self._lock:
            if self._initialized:
                return
            with self._connect() as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS rag_chunks (
                        id TEXT PRIMARY KEY,
                        collection TEXT NOT NULL,
                        source TEXT NOT NULL,
                        chunk_index INTEGER NOT NULL,
                        text TEXT NOT NULL,
                        embedding BLOB NOT NULL,
                        embedding_dim INTEGER NOT NULL,
                        metadata_json TEXT NOT NULL,
                        content_hash TEXT NOT NULL,
                        created_at REAL NOT NULL
                    )
                    """
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_rag_collection "
                    "ON rag_chunks(collection)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_rag_source "
                    "ON rag_chunks(collection, source)"
                )
            self._initialized = True

    def insert_chunks(
        self,
        chunks: Sequence[ChunkRecord],
        embeddings: Sequence[np.ndarray],
        replace_sources: bool = True,
    ) -> int:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")

        self._init()
        now = time.time()
        rows = []
        for chunk, embedding in zip(chunks, embeddings):
            vector = np.asarray(embedding, dtype=np.float32)
            content_hash = _sha256_text(chunk.text)
            row_id = _sha256_text(
                "\0".join(
                    [
                        chunk.collection,
                        chunk.source,
                        str(chunk.chunk_index),
                        content_hash,
                    ]
                )
            )
            metadata = {
                **chunk.metadata,
                "source": chunk.source,
                "chunk_index": chunk.chunk_index,
                "content_hash": content_hash,
            }
            rows.append(
                (
                    row_id,
                    chunk.collection,
                    chunk.source,
                    chunk.chunk_index,
                    chunk.text,
                    vector.tobytes(),
                    int(vector.shape[0]),
                    json.dumps(metadata, ensure_ascii=False),
                    content_hash,
                    now,
                )
            )

        with self._lock:
            with self._connect() as conn:
                if replace_sources:
                    for collection, source in sorted(
                        {(chunk.collection, chunk.source) for chunk in chunks}
                    ):
                        conn.execute(
                            "DELETE FROM rag_chunks WHERE collection = ? AND source = ?",
                            (collection, source),
                        )
                conn.executemany(
                    """
                    INSERT OR REPLACE INTO rag_chunks (
                        id, collection, source, chunk_index, text, embedding,
                        embedding_dim, metadata_json, content_hash, created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    rows,
                )
        return len(rows)

    def search(
        self,
        collection: str,
        query_embedding: np.ndarray,
        top_k: int = DEFAULT_TOP_K,
        source_contains: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        self._init()
        query = np.asarray(query_embedding, dtype=np.float32)
        query = query / max(float(np.linalg.norm(query)), 1e-12)

        sql = (
            "SELECT id, source, chunk_index, text, embedding, embedding_dim, "
            "metadata_json FROM rag_chunks WHERE collection = ?"
        )
        params: List[Any] = [collection]
        if source_contains:
            sql += " AND source LIKE ?"
            params.append(f"%{source_contains}%")

        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()

        scored: List[Dict[str, Any]] = []
        for row in rows:
            dim = int(row["embedding_dim"])
            if dim != int(query.shape[0]):
                continue
            vector = np.frombuffer(row["embedding"], dtype=np.float32)
            score = float(np.dot(query, vector))
            scored.append(
                {
                    "id": row["id"],
                    "score": score,
                    "source": row["source"],
                    "chunk_index": int(row["chunk_index"]),
                    "text": row["text"],
                    "metadata": _loads_json(row["metadata_json"]),
                }
            )

        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored[: max(1, top_k)]

    def stats(self) -> Dict[str, Any]:
        self._init()
        with self._connect() as conn:
            total = conn.execute("SELECT COUNT(*) AS c FROM rag_chunks").fetchone()["c"]
            collections = conn.execute(
                """
                SELECT collection, COUNT(*) AS chunks, COUNT(DISTINCT source) AS sources
                FROM rag_chunks
                GROUP BY collection
                ORDER BY collection
                """
            ).fetchall()
        return {
            "db_path": str(self.db_path),
            "total_chunks": int(total),
            "collections": [
                {
                    "collection": row["collection"],
                    "chunks": int(row["chunks"]),
                    "sources": int(row["sources"]),
                }
                for row in collections
            ],
        }


embedder = LocalEmbeddingService()
vector_store = LocalVectorStore(DB_PATH)


@router.get("/v1/local_rag/health")
@router.get("/local_rag/health")
async def local_rag_health() -> Dict[str, Any]:
    default_model = resolve_embedding_model(DEFAULT_EMBED_MODEL)
    return {
        "status": "ok",
        "embedding_model": default_model,
        "embedding_backend": resolve_embedding_backend(default_model),
        "loaded_embedding_model": embedder.loaded_model,
        "embedding_backends": embedder.backend_status(),
        "embedding_model_presets": EMBED_MODEL_PRESETS,
        "document_root": str(DOCUMENT_ROOT),
        "store": vector_store.stats(),
    }


@router.post("/v1/local/embeddings")
@router.post("/local/embeddings")
async def local_embeddings(request: Request) -> Dict[str, Any]:
    data = await _read_json(request)
    raw_input = data.get("input")
    if isinstance(raw_input, str):
        inputs = [raw_input]
    elif isinstance(raw_input, list) and all(isinstance(item, str) for item in raw_input):
        inputs = raw_input
    else:
        raise HTTPException(
            status_code=400,
            detail="input must be a string or a list of strings",
        )

    input_type = data.get("input_type", "text")
    instruction = data.get("instruction", DEFAULT_QUERY_INSTRUCTION)
    embed_inputs = [
        _format_query_for_embedding(item, instruction) if input_type == "query" else item
        for item in inputs
    ]

    model_id = resolve_embedding_model(
        str(data.get("model", data.get("embedding_model", DEFAULT_EMBED_MODEL)))
    )
    backend = data.get("backend", data.get("embedding_backend"))
    try:
        vectors = await embedder.aembed_texts(
            embed_inputs,
            model_id=model_id,
            backend=backend,
            max_length=int(data.get("max_length", DEFAULT_EMBED_MAX_LENGTH)),
            batch_size=int(data.get("batch_size", DEFAULT_EMBED_BATCH_SIZE)),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return {
        "object": "list",
        "model": model_id,
        "embedding_backend": resolve_embedding_backend(model_id, backend),
        "data": [
            {
                "object": "embedding",
                "index": index,
                "embedding": vector.astype(float).tolist(),
            }
            for index, vector in enumerate(vectors)
        ],
        "usage": {"prompt_tokens": 0, "total_tokens": 0},
    }


@router.post("/v1/local_rag/ingest")
@router.post("/local_rag/ingest")
async def local_rag_ingest(request: Request) -> Dict[str, Any]:
    options, documents = await _parse_ingest_request(request)

    collection = str(options.get("collection", "default"))
    chunk_size = int(options.get("chunk_size", DEFAULT_CHUNK_SIZE))
    chunk_overlap = int(options.get("chunk_overlap", DEFAULT_CHUNK_OVERLAP))
    replace_sources = bool(options.get("replace_sources", True))
    model_id = resolve_embedding_model(str(options.get("embedding_model", DEFAULT_EMBED_MODEL)))
    embedding_backend = options.get("embedding_backend")
    try:
        resolved_backend = resolve_embedding_backend(model_id, embedding_backend)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    max_length = int(options.get("max_length", DEFAULT_EMBED_MAX_LENGTH))
    batch_size = int(options.get("batch_size", DEFAULT_EMBED_BATCH_SIZE))

    chunks: List[ChunkRecord] = []
    skipped: List[Dict[str, Any]] = []
    for document in documents:
        document_chunks = split_text(document.text, chunk_size, chunk_overlap)
        if not document_chunks:
            skipped.append({"source": document.source, "reason": "no text chunks"})
            continue
        for index, text in enumerate(document_chunks):
            chunks.append(
                ChunkRecord(
                    collection=collection,
                    source=document.source,
                    chunk_index=index,
                    text=text,
                    metadata={
                        **document.metadata,
                        "embedding_model": model_id,
                        "embedding_backend": resolved_backend,
                    },
                )
            )

    if not chunks:
        raise HTTPException(
            status_code=400,
            detail={"error": "No ingestible text chunks found", "skipped": skipped},
        )

    try:
        embeddings = await embedder.aembed_texts(
            [chunk.text for chunk in chunks],
            model_id=model_id,
            backend=embedding_backend,
            max_length=max_length,
            batch_size=batch_size,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    inserted = await asyncio.to_thread(
        vector_store.insert_chunks,
        chunks,
        embeddings,
        replace_sources,
    )

    return {
        "object": "local_rag.ingest",
        "status": "completed",
        "collection": collection,
        "embedding_model": model_id,
        "embedding_backend": resolved_backend,
        "sources": len({chunk.source for chunk in chunks}),
        "chunks_inserted": inserted,
        "skipped": skipped,
    }


@router.post("/v1/local_rag/search")
@router.post("/local_rag/search")
async def local_rag_search(request: Request) -> Dict[str, Any]:
    data = await _read_json(request)
    query = data.get("query")
    if not isinstance(query, str) or not query.strip():
        raise HTTPException(status_code=400, detail="query is required")

    embedding_model = resolve_embedding_model(str(data.get("embedding_model", DEFAULT_EMBED_MODEL)))
    embedding_backend = data.get("embedding_backend")
    results = await _search(
        query=query,
        collection=str(data.get("collection", "default")),
        top_k=int(data.get("top_k", DEFAULT_TOP_K)),
        source_contains=data.get("source_contains"),
        model_id=embedding_model,
        backend=embedding_backend,
        instruction=str(data.get("instruction", DEFAULT_QUERY_INSTRUCTION)),
        max_length=int(data.get("max_length", DEFAULT_EMBED_MAX_LENGTH)),
        batch_size=int(data.get("batch_size", DEFAULT_EMBED_BATCH_SIZE)),
    )

    return {
        "object": "local_rag.search",
        "query": query,
        "collection": str(data.get("collection", "default")),
        "embedding_model": embedding_model,
        "embedding_backend": resolve_embedding_backend(embedding_model, embedding_backend),
        "data": results,
    }


@router.post("/v1/local_rag/query")
@router.post("/local_rag/query")
async def local_rag_query(request: Request) -> Dict[str, Any]:
    data = await _read_json(request)
    if data.get("stream"):
        raise HTTPException(status_code=400, detail="stream=true is not supported yet")

    messages = data.get("messages")
    if not isinstance(messages, list) or not messages:
        raise HTTPException(status_code=400, detail="messages must be a non-empty list")

    query = data.get("query") or extract_last_user_text(messages)
    if not isinstance(query, str) or not query.strip():
        raise HTTPException(status_code=400, detail="Could not extract a query")

    embedding_model = resolve_embedding_model(str(data.get("embedding_model", DEFAULT_EMBED_MODEL)))
    embedding_backend = data.get("embedding_backend")
    search_results = await _search(
        query=query,
        collection=str(data.get("collection", "default")),
        top_k=int(data.get("top_k", DEFAULT_TOP_K)),
        source_contains=data.get("source_contains"),
        model_id=embedding_model,
        backend=embedding_backend,
        instruction=str(data.get("instruction", DEFAULT_QUERY_INSTRUCTION)),
        max_length=int(data.get("max_length", DEFAULT_EMBED_MAX_LENGTH)),
        batch_size=int(data.get("batch_size", DEFAULT_EMBED_BATCH_SIZE)),
    )

    context_message = build_context_message(
        search_results,
        max_context_chars=int(data.get("max_context_chars", DEFAULT_MAX_CONTEXT_CHARS)),
    )
    augmented_messages = inject_context_message(messages, context_message)

    model = str(data.get("model", DEFAULT_CHAT_MODEL))
    completion = await call_qwen_completion(model=model, messages=augmented_messages, data=data)
    completion_data = to_plain_data(completion)

    return {
        "object": "local_rag.query",
        "model": model,
        "query": query,
        "collection": str(data.get("collection", "default")),
        "embedding_model": embedding_model,
        "embedding_backend": resolve_embedding_backend(embedding_model, embedding_backend),
        "answer": extract_answer(completion_data),
        "search_results": search_results,
        "completion": completion_data,
    }


async def _search(
    query: str,
    collection: str,
    top_k: int,
    source_contains: Optional[str],
    model_id: str,
    backend: Optional[str],
    instruction: str,
    max_length: int,
    batch_size: int,
) -> List[Dict[str, Any]]:
    resolved_model = resolve_embedding_model(model_id)
    try:
        vectors = await embedder.aembed_texts(
            [_format_query_for_embedding(query, instruction)],
            model_id=resolved_model,
            backend=backend,
            max_length=max_length,
            batch_size=batch_size,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return await asyncio.to_thread(
        vector_store.search,
        collection,
        vectors[0],
        top_k,
        source_contains,
    )


async def call_qwen_completion(
    model: str,
    messages: List[Dict[str, Any]],
    data: Dict[str, Any],
) -> Any:
    kwargs: Dict[str, Any] = {
        "temperature": data.get("temperature"),
        "max_tokens": data.get("max_tokens"),
        "top_p": data.get("top_p"),
    }
    kwargs = {key: value for key, value in kwargs.items() if value is not None}

    try:
        from litellm.proxy.proxy_server import llm_router

        if llm_router is not None:
            return await llm_router.acompletion(model=model, messages=messages, **kwargs)
    except Exception as exc:
        logger.warning("LiteLLM router completion path failed: %s", exc)

    import litellm

    return await litellm.acompletion(
        model=data.get("direct_model", "openai/RedHatAI/Qwen3.6-35B-A3B-NVFP4"),
        messages=messages,
        api_base=data.get(
            "api_base",
            os.environ.get("QWEN_RAG_CHAT_API_BASE", "http://localhost:11112/v1"),
        ),
        api_key=data.get("api_key", os.environ.get("QWEN_RAG_CHAT_API_KEY", "none")),
        **kwargs,
    )


async def _parse_ingest_request(request: Request) -> tuple[Dict[str, Any], List[SourceDocument]]:
    content_type = request.headers.get("content-type", "")
    options: Dict[str, Any] = {}
    documents: List[SourceDocument] = []

    if "multipart/form-data" in content_type:
        form = await request.form()
        request_field = form.get("request")
        if request_field:
            options.update(json.loads(str(request_field)))

        for key in (
            "collection",
            "chunk_size",
            "chunk_overlap",
            "replace_sources",
            "embedding_model",
            "embedding_backend",
            "max_length",
            "batch_size",
        ):
            if key in form:
                options[key] = _coerce_form_value(form[key])

        for _, value in form.multi_items():
            if hasattr(value, "read") and getattr(value, "filename", None):
                content = await value.read()
                source = getattr(value, "filename", "upload")
                text = extract_text_from_bytes(
                    source,
                    content,
                    getattr(value, "content_type", None),
                )
                documents.append(
                    SourceDocument(
                        source=f"upload:{source}",
                        text=text,
                        metadata={
                            "filename": source,
                            "content_type": getattr(value, "content_type", None),
                            "size": len(content),
                        },
                    )
                )
    else:
        data = await _read_json(request)
        options.update(data)
        documents.extend(_documents_from_json(data))

    path_value = options.get("path")
    if path_value:
        documents.extend(load_documents_from_path(str(path_value)))

    if not documents:
        raise HTTPException(
            status_code=400,
            detail="Provide text, texts, documents, path, or multipart file uploads",
        )

    return options, documents


def _documents_from_json(data: Dict[str, Any]) -> List[SourceDocument]:
    documents: List[SourceDocument] = []

    if isinstance(data.get("text"), str):
        source = str(data.get("source", "inline:text"))
        documents.append(
            SourceDocument(
                source=source,
                text=data["text"],
                metadata={"source_type": "inline"},
            )
        )

    raw_texts = data.get("texts")
    if isinstance(raw_texts, list):
        for index, item in enumerate(raw_texts):
            if isinstance(item, str):
                documents.append(
                    SourceDocument(
                        source=f"inline:texts:{index}",
                        text=item,
                        metadata={"source_type": "inline", "index": index},
                    )
                )
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                source = str(item.get("source", f"inline:texts:{index}"))
                metadata = dict(item.get("metadata") or {})
                metadata.update({"source_type": "inline", "index": index})
                documents.append(
                    SourceDocument(source=source, text=item["text"], metadata=metadata)
                )

    raw_documents = data.get("documents")
    if isinstance(raw_documents, list):
        for index, item in enumerate(raw_documents):
            if not isinstance(item, dict) or not isinstance(item.get("text"), str):
                continue
            source = str(item.get("source", f"inline:documents:{index}"))
            metadata = dict(item.get("metadata") or {})
            metadata.update({"source_type": "inline", "index": index})
            documents.append(
                SourceDocument(source=source, text=item["text"], metadata=metadata)
            )

    return documents


def load_documents_from_path(path_value: str) -> List[SourceDocument]:
    path = _resolve_document_path(path_value)
    if not path.exists():
        raise HTTPException(status_code=400, detail=f"Path does not exist: {path}")

    files = [path] if path.is_file() else list(_iter_document_files(path))
    documents: List[SourceDocument] = []
    for file_path in files:
        if file_path.stat().st_size > MAX_FILE_BYTES:
            logger.warning("Skipping oversized RAG file: %s", file_path)
            continue
        try:
            content = file_path.read_bytes()
            text = extract_text_from_bytes(file_path.name, content, None)
        except Exception as exc:
            logger.warning("Skipping unreadable RAG file %s: %s", file_path, exc)
            continue
        rel_source = _display_source(file_path)
        documents.append(
            SourceDocument(
                source=rel_source,
                text=text,
                metadata={
                    "source_type": "path",
                    "path": str(file_path),
                    "size": len(content),
                    "suffix": file_path.suffix.lower(),
                },
            )
        )
    return documents


def _resolve_document_path(path_value: str) -> Path:
    raw_path = Path(path_value).expanduser()
    if not raw_path.is_absolute():
        raw_path = DOCUMENT_ROOT / raw_path
    path = raw_path.resolve()

    allow_outside = os.environ.get("QWEN_RAG_ALLOW_OUTSIDE_ROOT", "0") == "1"
    if not allow_outside and not _is_relative_to(path, DOCUMENT_ROOT):
        raise HTTPException(
            status_code=403,
            detail=(
                f"Path ingestion is limited to {DOCUMENT_ROOT}. "
                "Set QWEN_RAG_ALLOW_OUTSIDE_ROOT=1 to allow other paths."
            ),
        )
    return path


def _iter_document_files(root: Path) -> List[Path]:
    files: List[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part.startswith(".") for part in path.relative_to(root).parts):
            continue
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        files.append(path)
    return files


def extract_text_from_bytes(
    filename: str,
    content: bytes,
    content_type: Optional[str],
) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf" or content_type == "application/pdf":
        try:
            from pypdf import PdfReader
        except Exception as exc:
            raise RuntimeError("PDF ingestion requires pypdf: pip install pypdf") from exc
        reader = PdfReader(BytesIO(content))
        pages = [page.extract_text() or "" for page in reader.pages]
        text = "\n\n".join(page.strip() for page in pages if page.strip())
        if not text:
            raise RuntimeError("No extractable text found in PDF")
        return text

    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"Could not decode file as text: {filename}")


def split_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return []
    chunk_size = max(200, chunk_size)
    chunk_overlap = max(0, min(chunk_overlap, chunk_size // 2))

    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        if end < len(text):
            cut = max(
                text.rfind("\n\n", start + chunk_size // 2, end),
                text.rfind("\n", start + chunk_size // 2, end),
                text.rfind(". ", start + chunk_size // 2, end),
            )
            if cut > start:
                end = cut + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(end - chunk_overlap, start + 1)
    return chunks


def build_context_message(
    results: Sequence[Dict[str, Any]],
    max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
) -> Dict[str, str]:
    header = (
        "Use the retrieved local document context below to answer the user's question. "
        "If the context does not contain the answer, say that the local documents do "
        "not provide enough information. Cite sources using [1], [2], etc.\n\n"
    )
    parts = [header]
    used = len(header)
    for index, result in enumerate(results, start=1):
        block = (
            f"[{index}] source={result['source']} chunk={result['chunk_index']} "
            f"score={result['score']:.4f}\n{result['text'].strip()}\n\n"
        )
        if used + len(block) > max_context_chars:
            break
        parts.append(block)
        used += len(block)
    return {"role": "system", "content": "".join(parts).strip()}


def inject_context_message(
    messages: Sequence[Dict[str, Any]],
    context_message: Dict[str, str],
) -> List[Dict[str, Any]]:
    copied = [dict(message) for message in messages]
    if copied and copied[0].get("role") == "system":
        return [copied[0], context_message, *copied[1:]]
    return [context_message, *copied]


def extract_last_user_text(messages: Sequence[Dict[str, Any]]) -> Optional[str]:
    for message in reversed(messages):
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text_parts = [
                item.get("text", "")
                for item in content
                if isinstance(item, dict) and item.get("type") == "text"
            ]
            if text_parts:
                return "\n".join(text_parts)
    return None


def extract_answer(completion_data: Any) -> Optional[str]:
    try:
        return completion_data["choices"][0]["message"]["content"]
    except Exception:
        return None


def to_plain_data(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    try:
        return json.loads(json.dumps(value, default=str))
    except Exception:
        return str(value)


def _last_token_pool(last_hidden_states: Any, attention_mask: Any, torch: Any) -> Any:
    left_padding = bool((attention_mask[:, -1].sum() == attention_mask.shape[0]).item())
    if left_padding:
        return last_hidden_states[:, -1]

    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    batch_indices = torch.arange(batch_size, device=last_hidden_states.device)
    return last_hidden_states[batch_indices, sequence_lengths]


def _format_query_for_embedding(query: str, instruction: str) -> str:
    return f"Instruct: {instruction}\nQuery: {query}"


async def _read_json(request: Request) -> Dict[str, Any]:
    try:
        data = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid JSON body") from exc
    if not isinstance(data, dict):
        raise HTTPException(status_code=400, detail="JSON body must be an object")
    return data


def _coerce_form_value(value: Any) -> Any:
    text = str(value)
    if text.lower() in {"true", "false"}:
        return text.lower() == "true"
    try:
        return int(text)
    except ValueError:
        return text


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _loads_json(value: str) -> Dict[str, Any]:
    try:
        data = json.loads(value)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _display_source(path: Path) -> str:
    if _is_relative_to(path, DOCUMENT_ROOT):
        return str(path.relative_to(DOCUMENT_ROOT))
    if _is_relative_to(path, BASE_DIR):
        return str(path.relative_to(BASE_DIR))
    return str(path)


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def install(app: Any) -> None:
    """Attach local RAG endpoints to a FastAPI app once."""
    if getattr(app.state, "qwen_local_rag_installed", False):
        return
    app.include_router(router)
    app.state.qwen_local_rag_installed = True
    logger.info("Installed local Qwen RAG endpoints")


def install_on_litellm_proxy() -> Any:
    """Import the LiteLLM proxy app and attach local RAG routes."""
    from litellm.proxy.proxy_server import app

    install(app)
    return app
