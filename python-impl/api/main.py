"""
FastAPI入口 — 提供REST API + SSE流式响应
"""

from __future__ import annotations

import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from agents.supervisor import create_supervisor_graph
from memory.working_memory import WorkingMemory
from memory.short_term import ShortTermMemory
from memory.long_term import LongTermMemory
from mcp.mcp_server import MCPToolServer, create_default_tools
from tracing.otel_config import init_tracer, AgentMetrics

load_dotenv()


working_memory = WorkingMemory()
short_term_memory = ShortTermMemory(redis_url=os.getenv("REDIS_URL", "redis://localhost:6379/0"))
long_term_memory = LongTermMemory(index_path=os.getenv("FAISS_INDEX_PATH", "./vector_store/faiss_index"))
mcp_server = create_default_tools(MCPToolServer())
metrics = AgentMetrics()
graph = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global graph

    # 注意：otel_config 在导入时已把 OTEL_EXPORTER_OTLP_ENDPOINT 从环境
    # 变量中清除（避免 SDK 自动装配不可达的导出器），端点副本保存在
    # otel_config._OTLP_ENDPOINT_ENV。这里传None 表示"沿用模块内保存的值"，
    # 由 init_tracer 自行判断是否需要启用导出。
    init_tracer(
        service_name=os.getenv("OTEL_SERVICE_NAME", "smart-cs-multi-agent"),
        otlp_endpoint=None,
    )

    graph = create_supervisor_graph(
        working_memory=working_memory,
        short_term_memory=short_term_memory,
        long_term_memory=long_term_memory,
    )

    long_term_memory.add_document(
        content="我们的理财产品A年化收益率为3.5%-5.2%，投资期限为6个月至3年，最低投资金额10000元。注意：理财非存款，产品有风险，投资须谨慎。",
        source="product_faq.md",
    )
    long_term_memory.add_document(
        content="退款政策：用户在购买后7天内可申请无理由退款，超过7天需提供合理原因。退款将在3-5个工作日内原路退回。",
        source="refund_policy.md",
    )
    long_term_memory.add_document(
        content="开户流程：1.准备身份证原件 2.填写开户申请表 3.进行视频认证 4.设置交易密码 5.完成风险评估问卷。整个流程约需15-30分钟。",
        source="account_guide.md",
    )
    # 补充常见口语问法。字符二元组相似度只认字面重合，
    # 且二元组必须连续出现（「退货」能匹配，「退钱」与「退款」匹配不到）。
    # 因此文档里要用完整词组把常见问法串起来，让二字组合能够命中。
    long_term_memory.add_document(
        content="退款、退钱、退货、退款申请、怎么退款、怎么退钱、如何退货、"
        "钱什么时候能退回来、申请退钱流程——以上问法均指退款业务。"
        "统一处理规则：购买后7天内可申请无理由退款，超过7天需说明合理原因，"
        "款项在3-5个工作日内原路退回原支付账户。",
        source="refund_policy.md",
    )
    long_term_memory.add_document(
        content="开户、开户流程、办理开户、注册账户、开通账户、开户需要什么材料——"
        "以上问法均指开户业务。需准备身份证原件，填写开户申请表，"
        "完成视频认证与风险评估问卷，全程约15-30分钟。",
        source="account_guide.md",
    )

    yield


app = FastAPI(
    title="智能客服多Agent系统",
    description="基于LangGraph的Supervisor编排多Agent智能客服系统",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    user_id: str = "anonymous"
    session_id: str | None = None


class ChatResponse(BaseModel):
    response: str
    session_id: str
    intent: str
    compliance_passed: bool


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """主聊天接口"""
    if graph is None:
        raise HTTPException(status_code=503, detail="系统初始化中")

    session_id = request.session_id or str(uuid.uuid4())

    await short_term_memory.add_message(session_id, "user", request.message)

    from langchain_core.messages import HumanMessage

    initial_state = {
        "messages": [HumanMessage(content=request.message)],
        "user_id": request.user_id,
        "session_id": session_id,
        "intent": "",
        "sub_results": {},
        "compliance_passed": True,
        "final_response": "",
        "current_agent": "",
        "retry_count": 0,
    }

    config = {"configurable": {"thread_id": session_id}}

    try:
        result = await graph.ainvoke(initial_state, config=config)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")

    final_response = result.get("final_response", "系统处理异常，请稍后重试")

    await short_term_memory.add_message(session_id, "assistant", final_response)

    return ChatResponse(
        response=final_response,
        session_id=session_id,
        intent=result.get("intent", "unknown"),
        compliance_passed=result.get("compliance_passed", True),
    )


@app.get("/api/history/{session_id}")
async def get_history(session_id: str):
    """获取对话历史"""
    history = await short_term_memory.get_history(session_id)
    return {"session_id": session_id, "messages": history}


@app.get("/api/tools")
async def list_tools():
    """MCP工具发现接口"""
    return {"tools": mcp_server.list_tools()}


@app.post("/api/tools/call")
async def call_tool(request: dict):
    """MCP工具调用接口"""
    result = await mcp_server.call_tool(
        name=request.get("name", ""),
        arguments=request.get("arguments", {}),
    )
    return {
        "success": result.success,
        "result": result.result,
        "error": result.error,
        "duration_ms": result.duration_ms,
    }


@app.get("/api/metrics")
async def get_metrics():
    """获取系统指标"""
    return {
        "agent_metrics": metrics.get_summary(),
        "tool_call_log": mcp_server.get_call_log(last_n=20),
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "1.0.0"}


@app.get("/api/health")
async def api_health_check():
    """前端演示页用的健康检查（与 /health 等价，路径不同仅为前端调用方便）。"""
    return {"status": "healthy", "version": "1.0.0"}


@app.get("/")
async def demo_ui():
    """返回内置的聊天演示页。

    原项目只有 FastAPI 自动生成的 Swagger 文档，没有任何可视化界面，
    这里挂一个零依赖的单文件演示页，便于直观看到编排链路的效果。
    """
    page = Path(__file__).resolve().parent.parent / "demo-ui.html"
    if not page.exists():
        raise HTTPException(status_code=404, detail="demo-ui.html 不存在")
    return FileResponse(page, media_type="text/html; charset=utf-8")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=True,
    )
