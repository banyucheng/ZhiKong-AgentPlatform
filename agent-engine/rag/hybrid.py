# Hybrid Retrieval: BM25 稀疏召回 + RRF 融合
# dense 向量擅长语义相似，但对错误码/型号/编号这类精确关键词容易漏召回；
# BM25 基于词频做稀疏匹配，恰好补上这个短板。两路结果用 RRF 融合排名。
import logging
import time
from typing import List

import jieba
from rank_bm25 import BM25Okapi
from sqlalchemy import create_engine, text

from config import settings
from rag.vector_store import CONNECTION_STRING

logger = logging.getLogger("rag.hybrid")

_bm25_cache = {}  # collection_name -> {"bm25": 索引, "chunks": [...], "built_at": ts}


def _tokenize(text: str) -> List[str]:
    """jieba 分词 + 小写归一，同时覆盖中文与英文关键词。"""
    return [t.lower() for t in jieba.lcut(text or "") if t.strip()]


def _fetch_all_chunks(collection_name: str) -> list:
    """直查 langchain_community PGVector 的底层表，拉取该 collection 全部片段。"""
    engine = create_engine(CONNECTION_STRING)
    sql = text("""
        SELECT e.document AS content, e.cmetadata AS metadata
        FROM langchain_pg_embedding e
        JOIN langchain_pg_collection c ON e.collection_id = c.uuid
        WHERE c.name = :name
    """)
    with engine.connect() as conn:
        rows = conn.execute(sql, {"name": collection_name}).fetchall()
    return [{"content": r[0] or "", "metadata": r[1] or {}} for r in rows]


def get_bm25_index(collection_name: str = "agent_knowledge"):
    """构建/缓存 BM25 索引，超过 TTL 自动重建（覆盖 celery 进程写入新文档的场景）。"""
    cached = _bm25_cache.get(collection_name)
    if cached and time.time() - cached["built_at"] < settings.bm25_ttl:
        return cached

    chunks = _fetch_all_chunks(collection_name)
    if not chunks:
        return None
    corpus = [_tokenize(c["content"]) for c in chunks]
    _bm25_cache[collection_name] = {
        "bm25": BM25Okapi(corpus),
        "chunks": chunks,
        "built_at": time.time(),
    }
    logger.info("BM25 index built: %d chunks (collection=%s)", len(chunks), collection_name)
    return _bm25_cache[collection_name]


def invalidate_bm25(collection_name: str = "agent_knowledge") -> None:
    """新增文档后主动失效缓存（仅对同进程生效，跨进程靠 TTL）。"""
    _bm25_cache.pop(collection_name, None)


def bm25_search(query: str, top_k: int = 20, collection_name: str = "agent_knowledge") -> list:
    """BM25 稀疏检索，返回与 dense 通道相同的 chunk dict 结构。"""
    idx = get_bm25_index(collection_name)
    if not idx:
        return []
    tokens = _tokenize(query)
    if not tokens:
        return []
    scores = idx["bm25"].get_scores(tokens)
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [
        {**idx["chunks"][i], "score": float(scores[i])}
        for i in order
        if scores[i] > 0  # 无任何词命中的片段不计入
    ]


def rrf_fuse(dense: list, sparse: list, k: int = 60, top_k: int = 5) -> list:
    """Reciprocal Rank Fusion: score = Σ 1/(k + rank)，k=60 为论文推荐值。

    优点：只看排名不看分数量纲，dense 余弦分与 BM25 分天然不可比的问题被绕开。
    用片段正文做去重键（同一片段两路都召回时分数叠加，排名靠前）。
    """
    scores = {}
    for rank, c in enumerate(dense, start=1):
        scores[c["content"]] = scores.get(c["content"], 0.0) + 1.0 / (k + rank)
    for rank, c in enumerate(sparse, start=1):
        scores[c["content"]] = scores.get(c["content"], 0.0) + 1.0 / (k + rank)

    dedup = {c["content"]: c for c in dense + sparse}
    ranked = sorted(dedup.values(), key=lambda c: scores[c["content"]], reverse=True)
    return ranked[:top_k]
