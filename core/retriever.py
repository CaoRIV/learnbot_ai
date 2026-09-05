"""
检索器 —— 混合检索 + 递归检索策略

学习要点：
- 混合检索（Hybrid Search）结合语义检索和关键词检索的优势
- alpha 参数控制两者权重（0.7 = 70% 语义 + 30% 关键词）
- 递归检索通过多轮迭代，利用 LLM 改写查询获取更全面的信息
"""

import logging
from config import HYBRID_ALPHA, RETRIEVAL_TOP_K, RERANK_TOP_K, MAX_RETRIEVAL_ITERATIONS
from core.evidence import Citation, RetrievedEvidence
from core.vector_store import index_lock, vector_store
from core.bm25_index import bm25_manager
from core.embeddings import encode_query
from core.reranker import rerank_results
from features.web_search import check_serpapi_key, search_web


def hybrid_merge(semantic_results, bm25_results, alpha=None, metadata_by_id=None):
    """
    合并语义检索和 BM25 检索结果

    使用加权分数：语义分数 × alpha + BM25分数 × (1-alpha)

    Args:
        semantic_results: {'ids': [[...]], 'documents': [[...]], 'metadatas': [[...]]}
        bm25_results: [{'id': ..., 'score': ..., 'content': ...}]
        alpha: 语义检索权重

    Returns:
        排序后的 [(doc_id, {'score': ..., 'content': ..., 'metadata': ...})]
    """
    if alpha is None:
        alpha = HYBRID_ALPHA
    if metadata_by_id is None:
        metadata_by_id = vector_store.metadatas_map

    merged_dict = {}

    # 处理语义检索结果
    if (semantic_results and
            isinstance(semantic_results.get('documents'), list) and len(semantic_results['documents']) > 0 and
            isinstance(semantic_results.get('metadatas'), list) and len(semantic_results['metadatas']) > 0 and
            isinstance(semantic_results.get('ids'), list) and len(semantic_results['ids']) > 0 and
            isinstance(semantic_results['documents'][0], list) and
            len(semantic_results['documents'][0]) == len(semantic_results['metadatas'][0]) == len(
                semantic_results['ids'][0])):
        num_results = len(semantic_results['documents'][0])
        for i, (doc_id, doc, meta) in enumerate(
                zip(semantic_results['ids'][0], semantic_results['documents'][0], semantic_results['metadatas'][0])):
            score = 1.0 - (i / max(1, num_results))
            merged_dict[doc_id] = {'score': alpha * score, 'content': doc, 'metadata': meta}
    else:
        logging.warning("Kết quả truy xuất ngữ nghĩa trống hoặc sai định dạng")

    # 处理 BM25 结果
    if not bm25_results:
        return sorted(merged_dict.items(), key=lambda x: x[1]['score'], reverse=True)

    valid_scores = [r['score'] for r in bm25_results if isinstance(r, dict) and 'score' in r]
    max_bm25 = max(valid_scores) if valid_scores else 1.0

    for result in bm25_results:
        if not (isinstance(result, dict) and 'id' in result and 'score' in result and 'content' in result):
            continue
        doc_id = result['id']
        norm_score = result['score'] / max_bm25 if max_bm25 > 0 else 0

        if doc_id in merged_dict:
            merged_dict[doc_id]['score'] += (1 - alpha) * norm_score
        else:
            metadata = metadata_by_id.get(doc_id, {})
            merged_dict[doc_id] = {
                'score': (1 - alpha) * norm_score,
                'content': result['content'], 'metadata': metadata
            }

    return sorted(merged_dict.items(), key=lambda x: x[1]['score'], reverse=True)


def _recursive_retrieval_with_scores(
    initial_query,
    max_iterations=None,
    enable_web_search=False,
    model_choice=None,
):
    """
    递归检索与查询优化

    流程：1.语义+BM25检索 → 2.混合排序 → 3.重排序 → 4.LLM判断是否改写query继续

    Returns:
        (all_contexts, all_doc_ids, all_metadata, all_scores)
    """
    if max_iterations is None:
        max_iterations = MAX_RETRIEVAL_ITERATIONS

    query = initial_query
    all_contexts, all_doc_ids, all_metadata, all_scores = [], [], [], []
    seen_web_sources = set()

    for i in range(max_iterations):
        logging.info(
            "Truy xuất đệ quy %s/%s, truy vấn hiện tại: %s",
            i + 1,
            max_iterations,
            query,
        )

        # 网络搜索补充
        web_texts = []
        if enable_web_search and check_serpapi_key():
            try:
                for res in search_web(query):
                    title = res.get('title') or ''
                    url = res.get('url') or ''
                    snippet = res.get('snippet') or ''
                    web_texts.append(f"Tiêu đề: {title}\nTóm tắt: {snippet}")
                    source_key = url or f"{title}\n{snippet}"
                    if snippet and source_key not in seen_web_sources:
                        seen_web_sources.add(source_key)
                        all_contexts.append(snippet)
                        all_doc_ids.append(f"web:{source_key}")
                        all_metadata.append({
                            'source': 'web',
                            'title': title,
                            'url': url,
                            'timestamp': res.get('timestamp'),
                        })
                        all_scores.append(None)
            except Exception as e:
                logging.error("Tìm kiếm web gặp lỗi: %s", e)

        # 语义检索
        query_embedding = encode_query(query)
        with index_lock:
            sem_docs, sem_ids, sem_metas = vector_store.search(
                query_embedding,
                k=RETRIEVAL_TOP_K,
            )
            prepared = {
                "ids": [sem_ids],
                "documents": [sem_docs],
                "metadatas": [sem_metas],
            }
            bm25_res = (
                bm25_manager.search(query, top_k=RETRIEVAL_TOP_K)
                if bm25_manager.bm25_index
                else []
            )
            metadata_by_id = dict(vector_store.metadatas_map)

        # 混合排序 → 重排序
        hybrid = hybrid_merge(
            prepared,
            bm25_res,
            metadata_by_id=metadata_by_id,
        )
        ids_iter, docs_iter, meta_iter = [], [], []
        for doc_id, data in hybrid[:RETRIEVAL_TOP_K]:
            ids_iter.append(doc_id)
            docs_iter.append(data['content'])
            meta_iter.append(data['metadata'])

        if docs_iter:
            try:
                reranked = rerank_results(query, docs_iter, ids_iter, meta_iter, top_k=RERANK_TOP_K)
            except Exception as e:
                logging.error("Xếp hạng lại kết quả thất bại: %s", e)
                reranked = [(did, {'content': d, 'metadata': m, 'score': 1.0})
                            for did, d, m in zip(ids_iter, docs_iter, meta_iter)]
        else:
            reranked = []

        # 整合结果
        current_contexts = web_texts[:]
        for doc_id, data in reranked:
            if doc_id not in all_doc_ids:
                all_doc_ids.append(doc_id)
                all_contexts.append(data['content'])
                all_metadata.append(data['metadata'])
                all_scores.append(data.get('score'))
            current_contexts.append(data['content'])

        if i == max_iterations - 1:
            break

        # LLM 判断是否需要继续
        if current_contexts:
            summary = "\n".join(current_contexts[:3])
            prompt = f"""Bạn là trợ lý tối ưu truy vấn. Hãy xác định có cần tạo truy vấn mới hay không.

[Câu hỏi ban đầu]
{initial_query}

[Tóm tắt kết quả truy xuất]
{summary}

Yêu cầu:
1. Nếu thông tin đã đủ, chỉ trả lời: KHÔNG CẦN TRUY VẤN THÊM
2. Nếu chưa đủ, chỉ trả về một truy vấn mới chính xác hơn
"""
            try:
                from core.generator import call_llm_simple
                next_query = call_llm_simple(prompt, model_choice)
                if "KHÔNG CẦN" in next_query.upper():
                    logging.info("LLM xác định không cần truy vấn thêm")
                    break
                if len(next_query) > 100:
                    logging.warning("Truy vấn do LLM tạo quá dài nên bị bỏ qua")
                    break
                query = next_query
                logging.info("Đã tạo truy vấn cho vòng tiếp theo: %s", query)
            except Exception as e:
                logging.error("Không thể tạo truy vấn mới: %s", e)
                break
        else:
            break

    return all_contexts, all_doc_ids, all_metadata, all_scores


def recursive_retrieval(initial_query, max_iterations=None, enable_web_search=False, model_choice=None):
    """API retrieval cũ, giữ nguyên tuple ba phần để tương thích ngược."""
    contexts, doc_ids, metadata, _scores = _recursive_retrieval_with_scores(
        initial_query=initial_query,
        max_iterations=max_iterations,
        enable_web_search=enable_web_search,
        model_choice=model_choice,
    )
    return contexts, doc_ids, metadata


def retrieve_evidence(initial_query, max_iterations=None, enable_web_search=False, model_choice=None):
    """Truy xuất các phân đoạn kèm citation chuẩn hóa từ metadata nguồn."""
    contexts, doc_ids, metadata_list, scores = _recursive_retrieval_with_scores(
        initial_query=initial_query,
        max_iterations=max_iterations,
        enable_web_search=enable_web_search,
        model_choice=model_choice,
    )
    evidence = []
    for content, doc_id, metadata, score in zip(
        contexts,
        doc_ids,
        metadata_list,
        scores,
    ):
        item_metadata = dict(metadata or {})
        is_web = item_metadata.get("source") == "web"
        if is_web:
            document = (
                item_metadata.get("title")
                or item_metadata.get("url")
                or "Nguồn web"
            )
        else:
            document = item_metadata.get("source") or "Không rõ nguồn"
        citation = Citation(
            document=document,
            page=None if is_web else item_metadata.get("page"),
            chunk_id=doc_id,
            score=score,
            source_type="web" if is_web else "document",
            url=item_metadata.get("url") if is_web else None,
        )
        evidence.append(
            RetrievedEvidence(
                content=content,
                citation=citation,
                metadata=item_metadata,
            )
        )
    return evidence
