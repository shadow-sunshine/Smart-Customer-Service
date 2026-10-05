"""
Supervisor编排Agent — 中央协调者
负责接收用户请求，根据意图路由到对应子Agent，汇总结果返回。
采用LangGraph StateGraph实现，支持并行调度和Human-in-the-Loop断点。
"""

from __future__ import annotations

import operator
import os
import re
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

from agents.intent_router import IntentRouterAgent
from agents.knowledge_rag import KnowledgeRAGAgent
from agents.ticket_handler import TicketHandlerAgent
from agents.compliance_checker import ComplianceCheckerAgent
from agents.executor import AgentExecutor
from memory.working_memory import WorkingMemory
from memory.short_term import ShortTermMemory
from memory.long_term import LongTermMemory
from tracing.otel_config import trace_agent_call


# ─── 状态定义 ───

class AgentState(TypedDict):
    """Supervisor编排的全局状态"""
    messages: Annotated[list[BaseMessage], add_messages]
    user_id: str
    session_id: str
    intent: str
    sub_results: dict[str, Any]
    compliance_passed: bool
    final_response: str
    current_agent: str
    retry_count: int
    # 检索相关度得分（最高分）与是否命中知识库。
    # 低于阈值时走拒答分支，这两个字段用于观测拒答率。
    relevance_score: float
    knowledge_hit: bool


# ─── Supervisor节点 ───

SUPERVISOR_SYSTEM_PROMPT = """你是一个智能客服系统的Supervisor（主管编排Agent）。
你的职责是：
1. 分析用户意图，决定分发给哪个子Agent处理
2. 汇总子Agent的处理结果，生成最终回复
3. 确保所有回复都经过合规审查

可用的子Agent：
- intent_router: 意图识别和分类
- knowledge_rag: 知识库检索和回答
- ticket_handler: 工单创建和查询
- compliance_checker: 合规审查和敏感词检测

根据用户消息，决定下一步路由到哪个Agent。
"""


class SupervisorNode:
    """Supervisor决策节点"""

    def __init__(self, llm: ChatOpenAI, working_memory: WorkingMemory):
        self.llm = llm
        self.working_memory = working_memory

    @trace_agent_call("supervisor")
    async def route_decision(self, state: AgentState) -> AgentState:
        """分析用户意图，决定路由"""
        messages = state["messages"]
        session_id = state.get("session_id", "default")

        context = self.working_memory.get_context(session_id)

        routing_prompt = [
            SystemMessage(content=SUPERVISOR_SYSTEM_PROMPT),
            SystemMessage(content=f"当前工作记忆上下文: {context}"),
            *messages,
            HumanMessage(content=(
                "请分析用户的最新消息，返回应该路由到的Agent名称。"
                "只返回以下之一: knowledge_rag, ticket_handler, compliance_checker"
            )),
        ]

        response = await self.llm.ainvoke(routing_prompt)
        intent = response.content.strip().lower()

        # LLM 偶尔会带上标点或附加说明，做一次归一化再校验，
        # 避免因为输出格式微差就fallback 到默认值。
        intent = re.sub(r"[^a-z_]", "", intent)
        for valid in ("knowledge_rag", "ticket_handler", "compliance_checker"):
            if valid in intent:
                intent = valid
                break

        valid_intents = {"knowledge_rag", "ticket_handler", "compliance_checker"}
        if intent not in valid_intents:
            intent = "knowledge_rag"

        self.working_memory.update(session_id, {"last_intent": intent})

        return {
            **state,
            "intent": intent,
            "current_agent": "supervisor",
        }

    @trace_agent_call("supervisor_synthesize")
    async def synthesize_response(self, state: AgentState) -> AgentState:
        """汇总子Agent结果，生成最终回复"""
        sub_results = state.get("sub_results", {})
        compliance_passed = state.get("compliance_passed", True)

        if not compliance_passed:
            final_response = (
                "抱歉，您的请求涉及敏感内容，已转交人工客服处理。"
                "工单编号已自动生成，请留意后续通知。"
            )
        else:
            # sub_results 里混有字符串（可直接作为回复正文）和 dict
            #（如 compliance 的结构化审查结果），原实现对所有条目直接 join，
            # 遇到 dict 会抛 "expected str instance, dict found"。
            result_parts = []
            for result in sub_results.values():
                if isinstance(result, str) and result.strip():
                    result_parts.append(result)
                elif isinstance(result, dict):
                    # 结构化结果只取其文本字段，避免把元数据当回复正文
                    text = result.get("answer") or result.get("content") or result.get("text")
                    if isinstance(text, str) and text.strip():
                        result_parts.append(text)
            final_response = "\n\n".join(result_parts) if result_parts else "抱歉，暂时无法处理您的请求，请稍后重试。"

        return {
            **state,
            "final_response": final_response,
            "messages": [AIMessage(content=final_response)],
        }


# ─── 路由函数 ───

def route_to_agent(state: AgentState) -> str:
    """根据意图路由到对应Agent节点"""
    intent = state.get("intent", "knowledge_rag")
    route_map = {
        "knowledge_rag": "knowledge_rag",
        "ticket_handler": "ticket_handler",
        "compliance_checker": "compliance_check",
    }
    return route_map.get(intent, "knowledge_rag")


def should_check_compliance(state: AgentState) -> str:
    """所有回复都需经过合规审查"""
    return "compliance_check"


# ─── 构建Graph ───

def create_supervisor_graph(
    llm: ChatOpenAI | None = None,
    working_memory: WorkingMemory | None = None,
    short_term_memory: ShortTermMemory | None = None,
    long_term_memory: LongTermMemory | None = None,
    enable_checkpointing: bool = True,
    agent_timeout: float = 10.0,
) -> StateGraph:
    """
    构建Supervisor编排的多Agent StateGraph。

    这是整个系统的核心入口，将4个子Agent通过有向图连接起来，
    由Supervisor节点负责路由决策和结果汇总。

    Args:
        llm: 语言模型实例
        working_memory: 工作记忆
        short_term_memory: 短期记忆
        long_term_memory: 长期记忆
        enable_checkpointing: 是否启用检查点（支持断点恢复）
    """
    if llm is None:
        # 从环境变量读取模型配置，便于切换 OpenAI / DeepSeek 等不同 provider。
        # 原实现硬编码 gpt-4o，导致 .env 中的 MODEL_NAME / OPENAI_BASE_URL 完全不生效，
        # 换任何非 OpenAI 的模型都会直接 400。
        llm = ChatOpenAI(
            model=os.getenv("MODEL_NAME", "deepseek-flash"),
            base_url=os.getenv("OPENAI_BASE_URL") or None,
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0,
        )
    if working_memory is None:
        working_memory = WorkingMemory()

    supervisor = SupervisorNode(llm, working_memory)

    intent_router = IntentRouterAgent(llm)
    knowledge_agent = KnowledgeRAGAgent(llm, long_term_memory)
    ticket_agent = TicketHandlerAgent(llm)
    compliance_agent = ComplianceCheckerAgent(llm)

    graph = StateGraph(AgentState)

    # 所有业务节点统一经 AgentExecutor 包装，获得超时控制与降级能力。
    # 原实现直接挂 Agent 函数，任一 LLM 调用挂住都会导致整条链路阻塞。
    executor = AgentExecutor(timeout=agent_timeout)

    def _guarded(name: str, fn):
        """把 Agent 节点包一层超时与降级。返回普通函数给 add_node。"""

        async def _invoke(state: dict) -> dict:
            return await executor.run(name, lambda: fn(state), state)

        return _invoke

    graph.add_node("intent_router", _guarded("intent_router", intent_router.process))
    graph.add_node(
        "supervisor_route", _guarded("supervisor_route", supervisor.route_decision)
    )
    graph.add_node("knowledge_rag", _guarded("knowledge_rag", knowledge_agent.process))
    graph.add_node("ticket_handler", _guarded("ticket_handler", ticket_agent.process))
    graph.add_node(
        "compliance_check", _guarded("compliance_check", compliance_agent.process)
    )
    graph.add_node("synthesize", _guarded("synthesize", supervisor.synthesize_response))

    # 原实现实例化了 IntentRouterAgent 但从未 add_node，导致意图识别实际是
    # Supervisor 里一句 prompt 让 LLM 直接返回节点名（字符串匹配式路由）。
    # 这里真正挂上意图识别节点，先分类再路由，并保留 Supervisor 做兜底决策。
    graph.set_entry_point("intent_router")
    graph.add_edge("intent_router", "supervisor_route")

    graph.add_conditional_edges(
        "supervisor_route",
        route_to_agent,
        {
            "knowledge_rag": "knowledge_rag",
            "ticket_handler": "ticket_handler",
            "compliance_check": "compliance_check",
        },
    )

    graph.add_edge("knowledge_rag", "compliance_check")
    graph.add_edge("ticket_handler", "compliance_check")
    graph.add_edge("compliance_check", "synthesize")
    graph.add_edge("synthesize", END)

    checkpointer = MemorySaver() if enable_checkpointing else None
    compiled = graph.compile(checkpointer=checkpointer)

    return compiled
