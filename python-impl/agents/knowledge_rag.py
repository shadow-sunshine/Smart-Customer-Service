"""
知识检索Agent — RAG知识库问答
负责从向量数据库中检索相关文档，结合上下文生成准确回答。
实现完整的RAG流程：Query改写 → 向量检索 → 重排序 → 上下文注入 → 生成回答。
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from memory.long_term import LongTermMemory
from tracing.otel_config import trace_agent_call


RAG_SYSTEM_PROMPT = """你是一个专业的知识库问答Agent，负责根据检索到的文档回答用户问题。

回答规则：
1. 严格基于检索到的文档内容回答，不要编造信息
2. 如果文档中没有相关信息，明确告知用户并建议转人工
3. 回答要简洁专业，适合客服场景
4. 对于金融产品信息，必须标注"以上信息仅供参考，具体以合同条款为准"
5. 在回答末尾标注引用的文档来源

回答格式：
- 先直接回答用户问题
- 如有必要补充相关信息
- 金融场景需添加风险提示
"""

QUERY_REWRITE_PROMPT = """请将用户的口语化问题改写为更适合向量检索的查询语句。
保留核心语义，去除口语化表达，补充专业术语。
只返回改写后的查询，不要其他内容。

用户原始问题: {query}
"""

# 检索不命中时的标准回复。
# 明确告知"知识库无相关信息"而不是硬编，同时给出可执行的下一步。
NO_KNOWLEDGE_ANSWER = (
    "抱歉，知识库中暂时没有与您问题相关的信息。\n"
    "为了避免给您不准确的答复，建议您转接人工客服进一步确认。"
)


class KnowledgeRAGAgent:
    """知识检索Agent - 实现完整RAG流程"""

    def __init__(
        self,
        llm: ChatOpenAI,
        long_term_memory: LongTermMemory | None = None,
        relevance_threshold: float | None = None,
    ):
        self.llm = llm
        self.long_term_memory = long_term_memory or LongTermMemory()
        # 相似度门限交给 LongTermMemory 统一负责（见 long_term.search()），
        # 这里不再重复设置。
        #
        # 之前这里硬编码了 0.35，而 LongTermMemory 用的是 0.01，
        # 两处阈值不一致导致所有正常问题都被误判为「知识库外」而拒答
        # （实测域内问题分数只有 0.03~0.09，全部低于 0.35）。
        #
        # 门禁必须在检索层做，因为只有那里掌握每篇文档的实际得分，
        # 并且能在返回前就过滤掉零分文档——若等到这里再判断，
        # LLM 重排可能已经把零分文档排到了首位。
        #
        # 若确需覆盖阈值，请通过 RELEVANCE_THRESHOLD 环境变量，
        # 由 LongTermMemory 读取，保持单一数据源。
        self.relevance_threshold = (
            relevance_threshold
            if relevance_threshold is not None
            else self.long_term_memory.relevance_threshold
        )

    @trace_agent_call("rag_query_rewrite")
    async def rewrite_query(self, original_query: str) -> str:
        """Query改写：将口语化问题转为检索友好的查询"""
        messages = [
            HumanMessage(content=QUERY_REWRITE_PROMPT.format(query=original_query)),
        ]
        response = await self.llm.ainvoke(messages)
        return response.content.strip()

    @trace_agent_call("rag_retrieve")
    async def retrieve_documents(self, query: str, top_k: int = 5) -> list[dict]:
        """从向量数据库检索相关文档"""
        docs = self.long_term_memory.search(query, top_k=top_k)
        return docs

    @trace_agent_call("rag_rerank")
    async def rerank_documents(
        self, query: str, documents: list[dict], top_k: int = 3
    ) -> list[dict]:
        """对检索结果重排序，提升相关性"""
        if not documents:
            return []

        doc_summaries = "\n".join(
            f"[{i}] {doc.get('content', '')[:200]}"
            for i, doc in enumerate(documents)
        )

        messages = [
            SystemMessage(content="你是一个文档相关性排序专家。"),
            HumanMessage(content=(
                f"用户查询: {query}\n\n"
                f"候选文档:\n{doc_summaries}\n\n"
                f"请返回最相关的{top_k}个文档的索引号，用逗号分隔，如: 0,2,4"
            )),
        ]

        response = await self.llm.ainvoke(messages)

        try:
            indices = [int(i.strip()) for i in response.content.split(",")]
            reranked = [documents[i] for i in indices if i < len(documents)]
        except (ValueError, IndexError):
            reranked = documents[:top_k]

        return reranked

    @trace_agent_call("rag_generate")
    async def generate_answer(self, query: str, context_docs: list[dict]) -> str:
        """基于检索文档生成回答"""
        if not context_docs:
            return "抱歉，知识库中暂未找到与您问题相关的信息。建议您联系人工客服获取帮助。"

        context = "\n\n---\n\n".join(
            f"来源: {doc.get('source', '未知')}\n内容: {doc.get('content', '')}"
            for doc in context_docs
        )

        messages = [
            SystemMessage(content=RAG_SYSTEM_PROMPT),
            HumanMessage(content=(
                f"用户问题: {query}\n\n"
                f"检索到的参考文档:\n{context}"
            )),
        ]

        response = await self.llm.ainvoke(messages)
        return response.content

    @trace_agent_call("knowledge_rag_process")
    async def process(self, state: dict[str, Any]) -> dict[str, Any]:
        """
        完整RAG流程（作为Graph节点）：
        1. Query改写
        2. 向量检索
        3. 重排序
        4. 生成回答
        """
        messages = state.get("messages", [])
        if not messages:
            return state

        original_query = messages[-1].content

        rewritten_query = await self.rewrite_query(original_query)

        raw_docs = await self.retrieve_documents(rewritten_query, top_k=5)

        reranked_docs = await self.rerank_documents(rewritten_query, raw_docs, top_k=3)

        # 抗幻觉关键一步：相似度阈值门禁。
        # 检索总会返回 top-k 条文档，哪怕全都与问题无关。若不加阈值判断，
        # 模型会拿着不相关的文档硬编答案（实测：问"年会在哪办"会编造流程）。
        # 最高分低于阈值即判定为知识库外，直接拒答并建议转人工。
        top_score = reranked_docs[0].get("score", 0.0) if reranked_docs else 0.0
        if top_score < self.relevance_threshold:
            return {
                **state,
                "sub_results": {
                    **state.get("sub_results", {}),
                    "knowledge_rag": NO_KNOWLEDGE_ANSWER,
                },
                "relevance_score": top_score,
                "knowledge_hit": False,
            }

        answer = await self.generate_answer(original_query, reranked_docs)

        return {
            **state,
            "sub_results": {
                **state.get("sub_results", {}),
                "knowledge_rag": answer,
            },
            "relevance_score": top_score,
            "knowledge_hit": True,
        }
