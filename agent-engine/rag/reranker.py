# Reranker: bge-reranker-v2-m3 交叉编码器精排
# 与双塔(bi-encoder)不同，cross-encoder 把 (query, doc) 拼在一起过模型，
# 能捕捉逐词交互关系，精度更高但只能对少量候选打分，因此放在召回之后。
import logging
import os

from config import settings

logger = logging.getLogger("rag.reranker")

RERANK_CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "reranker-cache")

_reranker = None
_reranker_failed = False


def _get_reranker():
    """懒加载 + 进程内缓存。加载失败时降级为不重排（保留召回顺序）。"""
    global _reranker, _reranker_failed
    if _reranker is not None or _reranker_failed:
        return _reranker
    try:
        os.makedirs(RERANK_CACHE_DIR, exist_ok=True)
        from FlagEmbedding import FlagReranker
        _reranker = FlagReranker(
            settings.rerank_model,
            cache_dir=RERANK_CACHE_DIR,
            use_fp16=False,  # CPU 上 fp16 反而更慢
        )
        logger.info("Reranker loaded: %s", settings.rerank_model)
    except Exception as e:
        _reranker_failed = True
        logger.warning("Reranker unavailable, fallback to recall order: %s", e)
    return _reranker


def rerank(query: str, chunks: list, top_k: int = 5) -> list:
    """按 query 对候选片段重新打分排序，返回 top_k。

    chunks: [{"content": str, "metadata": dict}, ...]
    失败/未启用时返回召回顺序的前 top_k，保证主链路不中断。
    """
    if not chunks:
        return []
    reranker = _get_reranker()
    if reranker is None:
        return chunks[:top_k]

    pairs = [[query, c["content"]] for c in chunks]
    try:
        scores = reranker.compute_score(pairs, normalize=True)
        if isinstance(scores, (int, float)):  # 单条候选时 FlagEmbedding 返回 float
            scores = [scores]
    except Exception as e:
        logger.warning("Rerank failed, fallback to recall order: %s", e)
        return chunks[:top_k]

    ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
    logger.info("Rerank done: %d candidates -> top %d, best=%.4f", len(chunks), top_k, max(scores))
    return [c for c, _ in ranked[:top_k]]
