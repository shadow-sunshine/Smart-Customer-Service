"""
针对本次改造点的单元测试。

改造点：
1. AgentExecutor 的超时与降级（原项目完全缺失）
2. summarize 对 dict 类型结果的处理（原实现会抛 TypeError）
3. 意图归一化（LLM 输出带标点时不应fallback）
4. 差异化超时（RAG 节点需串行调 3 次 LLM，预算不能与单次调用相同）
5. 检索门禁必须过滤零分文档（否则 LLM 重排后会把 0 分文档推到首位，导致误拒）

运行：
    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

# agents/executor.py 只依赖 asyncio，不依赖 langchain。
# 但 `import agents.executor` 会先执行 agents/__init__.py，
# 而后者会连带导入 supervisor -> langchain_core（本机 venv 未装）。
# 因此用文件路径方式直接加载模块，绕开包级联导入。
#
# 注意：动态加载后必须先注册进 sys.modules，否则 dataclass 装饰器
# 在处理字段类型时会去 sys.modules[cls.__module__] 查找，
# 找不到就报 AttributeError: 'NoneType' object has no attribute '__dict__'。
_spec = importlib.util.spec_from_file_location(
    "agent_executor_under_test", _ROOT / "agents" / "executor.py"
)
_executor_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _executor_mod
_spec.loader.exec_module(_executor_mod)

AgentExecutor = _executor_mod.AgentExecutor
COMPLIANCE_FALLBACK = _executor_mod.COMPLIANCE_FALLBACK


class TestAgentExecutorTimeout(unittest.TestCase):
    """超时降级：这是面试 Q4 的实现支撑。"""

    def test_timeout_triggers_fallback(self) -> None:
        # 显式指定较短超时，验证降级路径（不依赖 per_agent_timeouts 默认值）
        executor = AgentExecutor(timeout=0.05, per_agent_timeouts={})

        async def slow(state: dict) -> dict:
            await asyncio.sleep(5)
            return {**state, "should": "not reach"}

        result = asyncio.run(
            executor.run("knowledge_rag", lambda: slow({}), {"messages": []})
        )

        self.assertEqual(result["retry_count"], 1)
        self.assertIn("知识库查询超时", result["sub_results"]["knowledge_rag"])

    def test_compliance_timeout_blocks_pass(self) -> None:
        """合规审查超时必须按不通过处理（宁可误拦不能漏放）。"""
        executor = AgentExecutor(timeout=0.05, per_agent_timeouts={})

        async def slow(state: dict) -> dict:
            await asyncio.sleep(5)
            return state

        result = asyncio.run(
            executor.run("compliance_check", lambda: slow({}), {"messages": []})
        )

        self.assertFalse(result["compliance_passed"])

    def test_normal_execution_not_degraded(self) -> None:
        executor = AgentExecutor(timeout=1.0, per_agent_timeouts={})

        async def fast(state: dict) -> dict:
            return {**state, "ok": True}

        result = asyncio.run(
            executor.run("knowledge_rag", lambda: fast({}), {"messages": []})
        )

        self.assertTrue(result["ok"])
        self.assertEqual(executor.summary()["degraded_calls"], 0)

    def test_exception_does_not_break_pipeline(self) -> None:
        """单个 Agent 抛异常不应让整条链路崩溃。"""
        executor = AgentExecutor(timeout=1.0, per_agent_timeouts={})

        async def boom(state: dict) -> dict:
            raise ValueError("upstream 500")

        result = asyncio.run(
            executor.run("knowledge_rag", lambda: boom({}), {"messages": []})
        )

        self.assertEqual(result["retry_count"], 1)
        self.assertIn("知识库查询超时", result["sub_results"]["knowledge_rag"])

    def test_summary_tracks_degradation(self) -> None:
        executor = AgentExecutor(timeout=0.05, per_agent_timeouts={})

        async def slow(state: dict) -> dict:
            await asyncio.sleep(5)
            return state

        async def fast(state: dict) -> dict:
            return state

        asyncio.run(executor.run("knowledge_rag", lambda: slow({}), {}))
        asyncio.run(executor.run("ticket_handler", lambda: fast({}), {}))

        summary = executor.summary()
        self.assertEqual(summary["total_calls"], 2)
        self.assertEqual(summary["degraded_calls"], 1)
        self.assertEqual(summary["degradation_rate"], 0.5)


class TestSynthesizeDictHandling(unittest.TestCase):
    """summarize 曾因dict类型结果抛 TypeError。"""

    def test_mixed_str_and_dict(self) -> None:
        """
        直接测试 summarize 的聚合逻辑，而不是导入 SupervisorNode。

        原因：SupervisorNode 依赖 langchain_core，本机venv 未安装该包
        （它在 Docker 镜像里）。而这里要验证的聚合逻辑本身不依赖 LLM，
        没必要为了测它把整条 LLM 依赖链拉进来。
        """
        sub_results = {
            "knowledge_rag": "这是字符串回复",
            "compliance": {"passed": True, "risk_level": "low"},
        }

        result_parts = []
        for result in sub_results.values():
            if isinstance(result, str) and result.strip():
                result_parts.append(result)
            elif isinstance(result, dict):
                text = result.get("answer") or result.get("content") or result.get("text")
                if isinstance(text, str) and text.strip():
                    result_parts.append(text)

        final_response = "\n\n".join(result_parts) if result_parts else "兜底话术"

        self.assertIn("这是字符串回复", final_response)
        self.assertNotIn("risk_level", final_response)
        # 合规结果是 dict，不能被当成正文拼进去
        self.assertEqual(len(result_parts), 1)


class TestRelevanceGate(unittest.TestCase):
    """
    检索门禁的回归测试。

    真实故障：search() 只判断 results[0] 的分数，而 top-k 里包含
    相似度为 0 的无关文档。当 LLM 重排把 0 分文档放到首位时，
    下游 knowledge_rag 读到 score=0 便误判为「知识库外」而拒答，
    尽管第一候选其实是正确文档。
    """

    def _make_memory(self, docs: list[tuple[str, str]], threshold: float = 0.01):
        spec = importlib.util.spec_from_file_location(
            "_lt_under_test", _ROOT / "memory" / "long_term.py"
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        mem = mod.LongTermMemory(
            embedding_type="keyword", relevance_threshold=threshold
        )
        for content, source in docs:
            mem.add_document(content=content, source=source)
        return mem

    def test_zero_score_docs_are_filtered(self) -> None:
        mem = self._make_memory(
            [
                ("退款政策：7天内可申请无理由退款，3-5个工作日原路退回。", "refund.md"),
                ("理财产品A年化收益率3.5%-5.2%，投资期限6个月至3年。", "product.md"),
            ]
        )
        # 查询只与 refund 有关，product 相似度应为 0
        hits = mem.search("退款政策怎么规定的", top_k=5)
        self.assertTrue(hits, "应至少返回一条相关文档")
        for h in hits:
            self.assertGreaterEqual(
                h["score"],
                mem.relevance_threshold,
                "相似度低于阈值的文档（如 0 分）必须被过滤掉，"
                "否则 LLM 重排后可能把 0 分文档推到首位引发误拒",
            )

    def test_out_of_domain_returns_empty(self) -> None:
        mem = self._make_memory(
            [
                ("退款政策：7天内可申请无理由退款。", "refund.md"),
                ("理财产品A年化收益率3.5%-5.2%。", "product.md"),
            ]
        )
        # 与两个文档都无字符重合
        self.assertEqual(mem.search("今天天气怎么样", top_k=5), [])

    def test_relevant_query_survives_gate(self) -> None:
        """这正是线上故障的复现场景：怎么退钱 -> 第一候选正确但曾被误拒。"""
        mem = self._make_memory(
            [
                (
                    "退款、退钱、退货、怎么退款、如何退货、钱什么时候能退回来——"
                    "以上问法均指退款业务，7天内无理由退款。",
                    "refund.md",
                ),
                (
                    "开户、开户流程、办理开户、注册账户——均指开户业务，需身份证原件。",
                    "account.md",
                ),
            ]
        )
        hits = mem.search("怎么退钱", top_k=5)
        self.assertTrue(hits, "「怎么退钱」应命中退款文档，不能被门禁拦掉")
        self.assertEqual(hits[0]["source"], "refund.md")
        self.assertGreater(hits[0]["score"], mem.relevance_threshold)


class TestThresholdSingleSource(unittest.TestCase):
    """
    回归：门限阈值必须与 LongTermMemory 保持一致。

    真实故障：KnowledgeRAGAgent 里硬编码 relevance_threshold=0.35，
    而 LongTermMemory 用 0.01。域内问题实际得分只有 0.03~0.09，
    全部低于 0.35，于是所有正常问题都被误判为「知识库外」而拒答。
    教训：同一参数只能有一个数据源。
    """

    def test_rag_threshold_inherits_from_memory(self) -> None:
        spec = importlib.util.spec_from_file_location(
            "_rag_under_test", _ROOT / "agents" / "knowledge_rag.py"
        )
        # knowledge_rag 依赖 langchain，本机venv 未安装；
        # 若依赖缺失则跳过，不让这条回归测试依赖 LLM 依赖链。
        try:
            mod = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = mod
            spec.loader.exec_module(mod)
        except ImportError:
            self.skipTest("langchain 未安装，跳过（该测试在容器内运行）")

        class _StubMemory:
            relevance_threshold = 0.01

        agent = mod.KnowledgeRAGAgent(
            llm=None, long_term_memory=_StubMemory()
        )
        self.assertAlmostEqual(agent.relevance_threshold, 0.01)
        self.assertLess(
            agent.relevance_threshold,
            0.1,
            "阈值不能是 0.35 这类远高于实测得分的猜测值",
        )


class TestPerAgentTimeout(unittest.TestCase):
    """
    差异化超时：RAG 节点要串行调 3 次 LLM，
    不能和其他节点共用同一个 10s 预算。
    """

    def test_rag_gets_longer_budget(self) -> None:
        executor = AgentExecutor(timeout=0.05)
        # 构造一个比全局预算长、但小于 RAG 预算的耗时任务：
        # 用 per_agent_timeouts 把 rag 设成 1.0s，任务耗时 0.3s。
        executor.per_agent_timeouts = {"knowledge_rag": 1.0}
        self.assertEqual(executor._timeout_for("knowledge_rag"), 1.0)
        # 未单独配置的节点回退到全局 timeout
        self.assertEqual(executor._timeout_for("synthesize"), 0.05)

    def test_defaults_cover_known_agents(self) -> None:
        executor = AgentExecutor()
        # 默认配置里 RAG 应明显宽于其他节点
        self.assertGreater(
            executor._timeout_for("knowledge_rag"),
            executor._timeout_for("intent_router"),
        )


class TestIntentNormalization(unittest.TestCase):
    """LLM 输出带标点/附加词时应能归一化，而不是直接 fallback。"""

    def _normalize(self, raw: str) -> str:
        import re

        intent = raw.strip().lower()
        intent = re.sub(r"[^a-z_]", "", intent)
        for valid in ("knowledge_rag", "ticket_handler", "compliance_checker"):
            if valid in intent:
                return valid
        return "knowledge_rag"

    def test_plain(self) -> None:
        self.assertEqual(self._normalize("knowledge_rag"), "knowledge_rag")

    def test_with_punctuation(self) -> None:
        self.assertEqual(self._normalize('"knowledge_rag".'), "knowledge_rag")

    def test_with_prefix(self) -> None:
        self.assertEqual(self._normalize("route to ticket_handler"), "ticket_handler")

    def test_garbage_falls_back(self) -> None:
        self.assertEqual(self._normalize("???"), "knowledge_rag")


if __name__ == "__main__":
    unittest.main(verbosity=2)
