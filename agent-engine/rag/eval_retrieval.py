# Retrieval Evaluation: 检索质量评测（recall@k / MRR）
# 用法:
#   python -m rag.eval_retrieval                  # 跑全部模式 (dense / hybrid / pipeline)
#   python -m rag.eval_retrieval --mode dense     # 只测纯向量基线
#   python -m rag.eval_retrieval --top-k 10
# 模式说明:
#   dense    = 纯向量检索（升级前基线）
#   hybrid   = dense + BM25 + RRF 融合（不加重排）
#   pipeline = hybrid + bge-reranker 重排（当前线上完整链路）
import argparse
import json
import os
import time

from config import settings
from rag.vector_store import similarity_search, get_vector_store

EVAL_DATA_PATH = os.path.join(os.path.dirname(__file__), "eval_data", "golden_qa.json")


def load_golden_set(path: str = None) -> list:
    with open(path or EVAL_DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    items = [it for it in data.get("items", []) if it.get("enabled", False)]
    if not items:
        print("⚠ 黄金集没有 enabled=true 的条目。")
        print("  请编辑 rag/eval_data/golden_qa.json：把示例条目替换成知识库真实覆盖的问题，")
        print("  并将 enabled 改为 true 后重跑。")
    return items


def _chunk_matches(chunk: dict, matchers: list) -> bool:
    """单个片段是否命中任一匹配器。匹配器格式见 golden_qa.json 的 _usage。"""
    content = chunk.get("content", "")
    meta = chunk.get("metadata", {}) or {}
    for m in matchers or []:
        kws = m.get("content_contains")
        if kws and all(k in content for k in kws):
            return True
        sub = m.get("metadata_contains")
        if sub and any(sub in str(v) for v in meta.values()):
            return True
    return False


# ---------- 三种检索模式（复用线上同一套组件） ----------

def _dense_chunks(query: str, top_k: int) -> list:
    docs = similarity_search(query, top_k)
    return [{"content": d.page_content, "metadata": d.metadata} for d in docs]


def run_mode(mode: str, question: str, top_k: int) -> list:
    if mode == "dense":
        return _dense_chunks(question, top_k)

    from rag import hybrid, reranker

    if mode == "hybrid":
        dense = _dense_chunks(question, top_k)
        sparse = hybrid.bm25_search(question, top_k)
        return hybrid.rrf_fuse(dense, sparse, top_k=top_k)

    if mode == "pipeline":
        from rag.retriever import retrieve_context
        return retrieve_context(question, top_k=top_k)

    raise ValueError(f"unknown mode: {mode}")


# ---------- 指标计算 ----------

def evaluate_mode(mode: str, items: list, top_k: int) -> dict:
    hits, reciprocal_ranks, elapsed = 0, [], 0.0
    detail = []
    for it in items:
        t0 = time.perf_counter()
        try:
            chunks = run_mode(mode, it["question"], top_k)
        except Exception as e:
            detail.append({"id": it["id"], "error": str(e)})
            continue
        elapsed += time.perf_counter() - t0

        hit_rank = next(
            (i + 1 for i, c in enumerate(chunks) if _chunk_matches(c, it.get("any_of"))),
            None,
        )
        if hit_rank:
            hits += 1
            reciprocal_ranks.append(1.0 / hit_rank)
        detail.append({
            "id": it["id"],
            "hit": bool(hit_rank),
            "first_hit_rank": hit_rank,
            "top1_preview": (chunks[0]["content"][:40] + "...") if chunks else "-",
        })

    n = len(items)
    return {
        "mode": mode,
        "recall@k": round(hits / n, 4) if n else 0.0,
        "mrr": round(sum(reciprocal_ranks) / n, 4) if n else 0.0,
        "avg_latency_ms": round(elapsed / n * 1000, 1) if n else 0.0,
        "detail": detail,
    }


def main():
    parser = argparse.ArgumentParser(description="RAG 检索评测")
    parser.add_argument("--mode", choices=["dense", "hybrid", "pipeline", "all"], default="all")
    parser.add_argument("--top-k", type=int, default=None, help="覆盖黄金集里的 top_k")
    args = parser.parse_args()

    items = load_golden_set()
    if not items:
        return

    # 确认知识库非空
    store = get_vector_store()
    try:
        n_docs = len(store.similarity_search("test", k=1))
        if n_docs == 0:
            print("⚠ 向量库为空（agent_knowledge），请先上传文档再评测。")
            return
    except Exception as e:
        print(f"⚠ 无法连接向量库: {e}")
        return

    top_k = args.top_k
    modes = ["dense", "hybrid", "pipeline"] if args.mode == "all" else [args.mode]
    print(f"\n评测设置: top_k={top_k}  golden={len(items)} 条  模式={modes}\n")

    results = [evaluate_mode(m, items, top_k) for m in modes]

    print(f"{'mode':<10}{'recall@k':<12}{'MRR':<10}{'avg ms':<10}")
    for r in results:
        print(f"{r['mode']:<10}{r['recall@k']:<12}{r['mrr']:<10}{r['avg_latency_ms']:<10}")

    print("\n未命中明细:")
    misses = [(it["id"], d) for r in results for it, d in zip(items, r["detail"]) if not d.get("hit")]
    if not misses:
        print("  (无，全部命中)")
    else:
        for qid, d in misses[:10]:
            print(f"  [{qid}] rank={d.get('first_hit_rank')}  top1: {d.get('top1_preview')}")


if __name__ == "__main__":
    main()
