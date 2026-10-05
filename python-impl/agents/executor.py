"""
Agent 节点的容错包装：超时控制 + 降级 + 告警。

为什么需要这个模块：
面试若追问「某个 Agent 超时了 Supervisor 怎么处理」，需要有真实实现支撑。
原项目所有 Agent 直接挂在 LangGraph 图上，没有任何超时控制——
一旦某个 LLM 调用挂住，整条链路会一直阻塞，用户侧表现为「转圈不动」。

本模块提供三层防御：
1. Agent 级超时：asyncio.wait_for 包裹，超时抛 TimeoutError
2. Supervisor 降级：按 Agent 职责返回不同的兜底话术
3. 告警记录：把超时事件写入结构化日志，便于事后排查

设计取舍：
合规审查超时按「不通过」处理（宁可误拦不能漏放），
因为金融场景下漏放的代价远大于误拦带来的等待成本。
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

# 各 Agent 超时后的降级话术。按职责区分，避免统一回复造成的误导。
FALLBACK_MESSAGES = {
    "knowledge_rag": "抱歉，知识库查询超时了，您的问题我已经记录，会尽快为您核实。",
    "ticket_handler": "工单正在提交中，请稍后通过短信或站内信查看处理进度。",
    "intent_router": "抱歉，暂时无法识别您的需求，已为您转接人工客服。",
    "supervisor": "系统处理遇到异常，已为您转接人工客服。",
}

# 合规审查是底线环节，超时时不允许放行
COMPLIANCE_FALLBACK = "该回复需要人工复核后才能发送，已为您转接人工客服处理。"

DEFAULT_TIMEOUT_SECONDS = 10.0


@dataclass
class AgentRunResult:
    """记录一次 Agent 执行的耗时与是否降级，用于可观测与告警。"""

    agent: str
    duration_ms: int
    degraded: bool = False
    reason: str | None = None

    def log_line(self) -> str:
        flag = "DEGRADED" if self.degraded else "OK"
        detail = f" reason={self.reason}" if self.reason else ""
        return f"[agent_run] agent={self.agent} flag={flag} duration_ms={self.duration_ms}{detail}"


class AgentExecutor:
    """
    包裹 Agent 节点，提供超时与降级能力。

    用法：
        executor = AgentExecutor(timeout=10.0)
        state = await executor.run(
            "knowledge_rag",
            lambda: knowledge_agent.process(state),
            state,
        )
    """

    def __init__(self, timeout: float = DEFAULT_TIMEOUT_SECONDS) -> None:
        self.timeout = timeout
        self.runs: list[AgentRunResult] = []

    @staticmethod
    def _fallback_for(agent_name: str) -> str:
        if agent_name == "compliance_check":
            return COMPLIANCE_FALLBACK
        return FALLBACK_MESSAGES.get(
            agent_name, "系统处理遇到异常，已为您转接人工客服。"
        )

    async def run(
        self,
        agent_name: str,
        invoke: Callable[[], Awaitable[dict[str, Any]]],
        state: dict[str, Any],
    ) -> dict[str, Any]:
        """
        执行一个 Agent 节点。

        超时或异常时不抛出，而是返回带降级内容的 state，
        保证整条链路不会因为单个节点失败而中断。
        """
        started = time.perf_counter()
        try:
            result = await asyncio.wait_for(invoke(), timeout=self.timeout)
            elapsed = int((time.perf_counter() - started) * 1000)
            self.runs.append(AgentRunResult(agent=agent_name, duration_ms=elapsed))
            return result
        except asyncio.TimeoutError:
            elapsed = int((time.perf_counter() - started) * 1000)
            reason = f"timeout_after_{self.timeout}s"
            self.runs.append(
                AgentRunResult(
                    agent=agent_name,
                    duration_ms=elapsed,
                    degraded=True,
                    reason=reason,
                )
            )
            return self._degraded_state(state, agent_name, reason)
        except Exception as exc:  # noqa: BLE001 - 单点失败不应拖垮整条链路
            elapsed = int((time.perf_counter() - started) * 1000)
            reason = f"{type(exc).__name__}: {exc}"
            self.runs.append(
                AgentRunResult(
                    agent=agent_name,
                    duration_ms=elapsed,
                    degraded=True,
                    reason=reason,
                )
            )
            return self._degraded_state(state, agent_name, reason)

    def _degraded_state(
        self, state: dict[str, Any], agent_name: str, reason: str
    ) -> dict[str, Any]:
        """构造降级后的 state：写入兜底话术并标记失败。"""
        message = self._fallback_for(agent_name)
        # 合规降级直接置为不通过，后续 synthesize 会走转人工分支
        compliance_passed = state.get("compliance_passed", True)
        if agent_name == "compliance_check":
            compliance_passed = False

        sub_results = dict(state.get("sub_results", {}))
        sub_results[agent_name] = message

        new_state: dict[str, Any] = {
            **state,
            "sub_results": sub_results,
            "compliance_passed": compliance_passed,
            "retry_count": state.get("retry_count", 0) + 1,
        }

        if agent_name == "synthesize":
            new_state["final_response"] = message
            # 惰性导入：让本模块不硬依赖 langchain，便于独立测试
            try:
                from langchain_core.messages import AIMessage

                new_state["messages"] = list(state.get("messages", [])) + [
                    AIMessage(content=message)
                ]
            except ImportError:
                new_state["messages"] = list(state.get("messages", []))
        return new_state

    def summary(self) -> dict[str, Any]:
        """汇总各Agent 耗时与降级次数，供 /api/metrics 暴露。"""
        total = len(self.runs)
        degraded = sum(1 for r in self.runs if r.degraded)
        by_agent: dict[str, dict[str, Any]] = {}
        for r in self.runs:
            slot = by_agent.setdefault(
                r.agent, {"calls": 0, "degraded": 0, "total_ms": 0}
            )
            slot["calls"] += 1
            slot["total_ms"] += r.duration_ms
            if r.degraded:
                slot["degraded"] += 1
        for slot in by_agent.values():
            slot["avg_ms"] = slot["total_ms"] // max(slot["calls"], 1)
        return {
            "total_calls": total,
            "degraded_calls": degraded,
            "degradation_rate": round(degraded / total, 4) if total else 0.0,
            "by_agent": by_agent,
        }
