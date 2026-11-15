# RAG Retriever: 检索增强生成
# 检索管线: 双路召回(dense + BM25) -> RRF 融合 -> bge-reranker 精排 -> top_k
# 开关(enable_hybrid/enable_rerank)关闭或组件异常时自动退化为纯 dense 检索。
import logging

from config import settings
from rag.vector_store import similarity_search

logger = logging.getLogger("rag.retriever")


def _dense_chunks(query: str, top_k: int, collection_name: str) -> list:
    """dense 向量召回，统一转为 chunk dict 结构"""
    docs = similarity_search(query, top_k, collection_name)
    return [{"content": d.page_content, "metadata": d.metadata} for d in docs]


def retrieve_context(query: str, top_k: int = 5) -> list:
    """检索相关文档片段，返回 [{"content": ..., "metadata": ...}, ...]（对外契约不变）"""
    collection = "agent_knowledge"
    candidates = max(settings.retrieve_candidates, top_k)

    # 1. 召回：混合召回或纯 dense
    if settings.enable_hybrid:
        try:
            from rag import hybrid
            dense = _dense_chunks(query, candidates, collection)
            sparse = hybrid.bm25_search(query, candidates, collection)
            recalled = hybrid.rrf_fuse(dense, sparse, top_k=candidates)
            logger.info("Hybrid recall: dense=%d sparse=%d fused=%d", len(dense), len(sparse), len(recalled))
        except Exception as e:
            logger.warning("Hybrid recall failed, fallback to dense-only: %s", e)
            recalled = _dense_chunks(query, candidates, collection)
    else:
        recalled = _dense_chunks(query, candidates, collection)

    # 2. 精排：cross-encoder 重排
    if settings.enable_rerank:
        try:
            from rag.reranker import rerank
            return rerank(query, recalled, top_k=top_k)
        except Exception as e:
            logger.warning("Rerank failed, fallback to recall order: %s", e)

    return recalled[:top_k]


def build_rag_prompt(query: str, context_docs: list, system_prompt: str = None) -> str:
    """构建RAG提示词"""
    context = "\n\n---\n\n".join([d["content"] for d in context_docs])
    base = system_prompt or "你是一个智能助手,根据以下参考文档回答用户问题。如果文档中没有相关信息,请如实说明。"
    return f"""{base}

## 参考文档:
{context}

## 用户问题:
{query}

## 回答:"""
