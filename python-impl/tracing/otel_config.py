"""
全链路追踪 — OpenTelemetry集成
为每个Agent调用创建Span，记录延迟、Token消耗、路由决策等关键指标。
支持导出到Jaeger/Zipkin/LangSmith等后端。
"""

from __future__ import annotations

import functools
import os
import time
from typing import Any, Callable

# 必须在导入 opentelemetry SDK 之前处理：
# SDK 的 Metrics/MeterProvider 会在导入时自动读取
# OTEL_EXPORTER_OTLP_ENDPOINT 环境变量并自动装配 OTLP 导出器。
# 若该端点不可达，后台线程会持续重试 /v1/metrics，
# 这些等待发生在请求路径上，会拖慢响应甚至触发 Agent 超时。
#
# 因此这里先记下用户配置，再把环境变量清掉，
# 让 SDK 走 no-op 实现。显式传入 init_tracer 的endpoint 不受影响。
_OTLP_ENDPOINT_ENV = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "")
if os.environ.get("OTEL_SDK_DISABLED", "").lower() in ("1", "true", "yes"):
    _OTLP_ENDPOINT_ENV = ""
elif not _OTLP_ENDPOINT_ENV.strip():
    # 未配置端点时也要清掉空字符串，避免 SDK 反复尝试连空地址
    pass
else:
    os.environ.pop("OTEL_EXPORTER_OTLP_ENDPOINT", None)
# metrics 走独立的 OTEL_METRICS_EXPORTER，一并清掉
os.environ.pop("OTEL_METRICS_EXPORTER", None)
os.environ.pop("OTEL_EXPORTER_OTLP_METRICS_ENDPOINT", None)

try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    from opentelemetry.sdk.resources import Resource

    _HAS_OTEL = True
except ImportError:
    _HAS_OTEL = False


_tracer = None


def init_tracer(
    service_name: str = "smart-cs-multi-agent",
    otlp_endpoint: str | None = None,
) -> None:
    """
    初始化OpenTelemetry追踪器。

    Args:
        service_name: 服务名称
        otlp_endpoint: OTLP收集器地址。为空则不导出，链路追踪降级为
            本地记录（不产生网络请求）。

    为什么需要显式判断端点是否可用：
    OTLP 导出是异步批量进行的，当端点不可达时，BatchSpanProcessor
    会按退避策略反复重试（实测 0.9s / 1.8s / 3.7s 三次），
    每次都要等超时。这些等待发生在请求处理路径上，会吃掉
    Agent 的超时预算，导致本该正常返回的请求被判为超时。
    因此追踪后端不可用时应彻底关闭导出，而不是让它重试。
    """
    global _tracer

    if not _HAS_OTEL:
        return

    # 端点为空视为显式关闭导出
    # （OTEL_SDK_DISABLED=true 也可禁用）
    if os.getenv("OTEL_SDK_DISABLED", "").lower() in ("1", "true", "yes"):
        return
    # 模块导入时环境变量已被清掉，这里用保存下来的副本
    effective_endpoint = otlp_endpoint if otlp_endpoint is not None else _OTLP_ENDPOINT_ENV
    if not effective_endpoint or not effective_endpoint.strip():
        # 不配置导出端点：不注册任何 BatchSpanProcessor，
        # 避免后台线程做无意义的网络重试。
        _tracer = trace.get_tracer(service_name)
        return

    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)

    try:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        exporter = OTLPSpanExporter(endpoint=effective_endpoint)
    except ImportError:
        exporter = ConsoleSpanExporter()

    provider.add_span_processor(BatchSpanProcessor(exporter))
    # 避免重复设置 provider 导致启动告警：
    # "Overriding of current TracerProvider is not allowed"。
    # uvicorn --reload 或测试里init_tracer 可能被多次调用。
    try:
        trace.set_tracer_provider(provider)
    except Exception:
        pass
    _tracer = trace.get_tracer(service_name)


def get_tracer():
    """获取全局Tracer实例"""
    global _tracer
    if _tracer is None:
        if _HAS_OTEL:
            _tracer = trace.get_tracer("smart-cs-multi-agent")
        else:
            return None
    return _tracer


def trace_agent_call(agent_name: str) -> Callable:
    """
    Agent调用追踪装饰器。

    为每个Agent方法创建一个Span，记录：
    - agent.name: Agent名称
    - agent.duration_ms: 调用耗时
    - agent.input_size: 输入大小
    - agent.success: 是否成功

    用法：
        @trace_agent_call("knowledge_rag")
        async def process(self, state):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            tracer = get_tracer()

            if tracer is None:
                return await func(*args, **kwargs)

            span_name = f"agent.{agent_name}.{func.__name__}"

            with tracer.start_as_current_span(span_name) as span:
                span.set_attribute("agent.name", agent_name)
                span.set_attribute("agent.method", func.__name__)

                start_time = time.time()
                try:
                    result = await func(*args, **kwargs)
                    duration_ms = (time.time() - start_time) * 1000

                    span.set_attribute("agent.duration_ms", duration_ms)
                    span.set_attribute("agent.success", True)

                    if isinstance(result, dict):
                        span.set_attribute("agent.result_keys", str(list(result.keys())))

                    return result

                except Exception as e:
                    duration_ms = (time.time() - start_time) * 1000
                    span.set_attribute("agent.duration_ms", duration_ms)
                    span.set_attribute("agent.success", False)
                    span.set_attribute("agent.error", str(e))
                    span.record_exception(e)
                    raise

        return wrapper
    return decorator


class AgentMetrics:
    """Agent调用指标收集器"""

    def __init__(self):
        self._call_counts: dict[str, int] = {}
        self._total_duration: dict[str, float] = {}
        self._error_counts: dict[str, int] = {}

    def record_call(self, agent_name: str, duration_ms: float, success: bool):
        self._call_counts[agent_name] = self._call_counts.get(agent_name, 0) + 1
        self._total_duration[agent_name] = self._total_duration.get(agent_name, 0.0) + duration_ms
        if not success:
            self._error_counts[agent_name] = self._error_counts.get(agent_name, 0) + 1

    def get_summary(self) -> dict[str, Any]:
        summary = {}
        for agent_name in self._call_counts:
            calls = self._call_counts[agent_name]
            total_ms = self._total_duration[agent_name]
            errors = self._error_counts.get(agent_name, 0)
            summary[agent_name] = {
                "total_calls": calls,
                "avg_duration_ms": total_ms / calls if calls > 0 else 0,
                "error_rate": errors / calls if calls > 0 else 0,
            }
        return summary
