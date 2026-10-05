# Smart Customer Service

基于 LangGraph 的多Agent 智能客服系统，通过 Supervisor 编排模式协调意图路由、知识检索、工单处理与合规审查四类Agent，并配套一套可复现的检索质量评测体系。

仓库提供 Python、Java、Go 三套实现，**其中 Python 版经过完整的功能补强与量化评测，是唯一经过验证的可用版本**。

---

## 目录

- [项目简介](#项目简介)
- [设计目标](#设计目标)
- [系统架构](#系统架构)
- [核心能力](#核心能力)
- [检索质量评测](#检索质量评测)
- [技术栈](#技术栈)
- [目录结构](#目录结构)
- [安装与运行](#安装与运行)
- [配置项](#配置项)
- [API 说明](#api-说明)
- [评测集使用](#评测集使用)
- [常见问题](#常见问题)
- [当前限制](#当前限制)
- [许可证](#许可证)

---

## 项目简介

智能客服系统要解决的核心问题是**多环节串联**：一次用户提问往往同时涉及意图判断、知识查证、业务办理与合规检查。把这些环节塞进单个LLM 调用，会导致路由靠猜、回复无法追溯、且缺少统一的降级出口。

本项目采用**图编排**思路，把上述环节拆成独立节点，由 Supervisor 统一调度：

- 每个 Agent 只负责一件事，输入输出明确
- 所有回复**强制经过合规节点**，不设旁路
- 单个节点超时或异常时降级返回，不阻断整条链路
- 检索质量有标注数据集支撑，指标可复现

## 设计目标

| 目标 | 具体要求 |
|------|---------|
| 环节解耦 | 意图识别、检索、业务处理、合规审查互不依赖，可单独替换 |
| 行为可控 | 回复必须过合规门；检索无依据时拒答而非猜测 |
| 故障隔离 | 任一节点异常不导致整体不可用 |
| 效果可测 | 关键环节有量化指标，而非主观判断 |

---

## 系统架构

### 编排链路

```
                        ┌──────────────────────┐
      HTTP / WebSocket ─▶  IntentRouter        │ 意图分类 + 实体抽取
                        └──────────┬───────────┘
                                   ▼
                        ┌──────────────────────┐
                        │  Supervisor路由决策   │ 归一化 + 白名单校验
                        └──────────┬───────────┘
                    ┌──────────────┴──────────────┐
                    ▼                             ▼
        ┌───────────────────────┐    ┌───────────────────────┐
        │  KnowledgeRAG 知识检索 │    │  TicketHandler 工单处理│
        │  改写→检索→重排→生成   │    │  创建 / 更新 / 查询    │
        └───────────┬───────────┘    └───────────┬───────────┘
                    └──────────────┬──────────────┘
                                   ▼
                        ┌──────────────────────┐
                        │  ComplianceChecker   │ 规则引擎 → LLM 审查
                        │  敏感词 / PII / 风险分级│
                        └──────────┬───────────┘
                                   ▼
                        ┌──────────────────────┐
                        │  Supervisor 汇总回复   │ 标注引用来源
                        └──────────────────────┘
```

### 节点容错包装

所有业务节点统一经`AgentExecutor` 包装后才挂到图上：

```
Agent 节点 ──▶ AgentExecutor ──▶ asyncio.wait_for(10s)
                   │
                   ├─ 正常返回 ──▶ 记录耗时
                   ├─ 超时─────▶ 按职责返回降级话术
                   └─ 异常─────▶ 同上，不中断链路
```

合规节点超时按**不通过**处理，理由见[常见问题](#合规审查超时为什么不放行)。

### 记忆分层

| 层级 | 存储 | 生命周期 | 当前状态 |
|------|------|---------|---------|
| 工作记忆 | 进程内 dict + 线程锁 | 单次请求 | 生效，Supervisor 路由时读取 |
| 短期记忆 | Redis（TTL 30 分钟） | 会话级 | **仅写入，未接入读取** |
| 长期记忆 | FAISS 向量索引 | 持久 | 生效，承载知识库检索 |

多轮上下文目前由 LangGraph `MemorySaver` 检查点承载，短期记忆层的 `get_context_window()` 尚未被图节点调用。

---

## 核心能力

### 1. 意图识别

`agents/intent_router.py` 输出意图分类、置信度与关键实体，置信度可用于低置信度转人工。

### 2. 知识检索（RAG）

`agents/knowledge_rag.py` 的完整流程：

```
用户问题
  → Query 改写（去口语化，补专业术语）
  → 向量检索Top-5
  → LLM 重排序 → Top-3
  → 相似度门禁（低于阈值直接拒答）
  → 上下文注入 → LLM 生成
  → 附引用来源
```

门禁是抗幻觉的关键：检索总会返回 Top-K，即使全不相关，缺少门禁时模型会拿无关文档编造答案。

### 3. 可插拔Embedding

`memory/embedding.py` 提供三种实现，通过 `EMBEDDING_TYPE` 切换：

| 类型 | 说明 | 依赖 |
|------|------|------|
| `keyword` | 字符二元组 + Jaccard 相似度 | 无 |
| `openai` | OpenAI 兼容 embedding，可接 DeepSeek | API Key |
| `fake` | sha256 伪随机向量，仅用于回归对照 | 无 |

`keyword` 作为默认值，因为它离线可跑、无需密钥，且在评测集上表现最好。

### 4. MCP 工具协议

`mcp/mcp_server.py` 实现工具注册、发现、调用与 JSON-RPC 2.0 分发：

| 工具 | 作用 |
|------|------|
| `order_query` | 查询订单状态 |
| `knowledge_search` | 知识库搜索 |
| `ticket_create` | 创建工单 |
| `risk_check` | 风控校验 |

工具通过装饰器注册，`input_schema` 用 JSON Schema 描述入参。

### 5. 两阶段合规审查

`agents/compliance_checker.py`：

- **第一阶段（规则引擎）**：正则匹配手机号、身份证、银行卡、邮箱；关键词匹配违规用语
- **第二阶段（LLM 审查）**：处理隐晦违规表达，如"风险极低"这类规则难以覆盖的表述
- **风险分级**：violation 按 low / medium / high / critical 合并，取最高级别
- **PII 脱敏**：命中的敏感信息自动掩码

### 6. 全链路追踪

`tracing/otel_config.py` 提供 `trace_agent_call` 装饰器，为每个 Agent 记录 Span。指标通过 `/api/metrics` 暴露。

### 7. 可视化演示页

`demo-ui.html` 为零依赖单文件页面（原生 JS，无构建步骤），直接调用 `/api/chat`，展示意图路由、耗时、合规判定与引用来源。

---

## 检索质量评测

### 为什么需要评测

原实现的 embedding 用 sha256 当随机种子生成伪随机向量，不含任何语义。在 17 条标注 query 上实测Recall@1 仅 38.5%，且知识库外问题 100% 被硬答。

**任何检索策略的改动都应该用评测验证，而不是凭感觉。**

### 评测集构成

`python-impl/eval/eval_set.json`，17 条人工标注 query：

| 类别 | 条数 | 说明 |
|------|------|------|
| `in_domain` | 10 | query 关键词与文档高度重合 |
| `in_domain_colloquial` | 2 | 口语化，如「钱什么时候能退回来」 |
| `in_domain_similar` | 1 | 同义改写，如「我想申请退货怎么办」 |
| `out_of_domain` | 3 | 业务外，如「客服电话号码是多少」 |
| `out_of_domain_chitchat` | 1 | 闲聊，如「今天天气怎么样」 |

**域内域外必须分开评估。** 只统计域内会掩盖幻觉问题——域外才是幻觉高发区。

### 指标定义

| 指标 | 含义 |
|------|------|
| Recall@1 | 正确文档排在第一位的比例 |
| MRR | 正确文档排名的倒数均值 |
| OOD拒答率 | 域外问题被正确拒答的比例 |

### 实测结果

| 指标 | `fake`（改造前） | `keyword` |
|------|------------------|-----------|
| Recall@1 | 38.5% | **100%** |
| MRR | 0.654 | **1.000** |
| 口语化问题 | 0/2 | **2/2** |
| 相似问法 | 0/1 | **1/1** |
| 域外拒答率（无门禁） | 0% | 0% |
| 域外拒答率（门禁 0.01） | — | **100%** |

### 相似度门限的确定

门限值不是凭感觉设的，来自分数分布实测：

- 域外 query 最高分：**0.0000**
- 域内query 最低分：**0.0149**

取 `0.01` 落在两者之间，因此召回率零损失的前提下获得 100% 拒答率。

对照实验：若用 `fake` 后端，门限无法区分域内域外（门限 0.1 时域内召回率也跌到 0%），这也反证无语义 embedding 的问题。

详细分析见 [`python-impl/docs/eval_report.md`](python-impl/docs/eval_report.md)。

---

## 技术栈

| 层次 | Python 版 | Java 版 | Go 版 |
|------|-----------|---------|--------|
| 编排框架 | LangGraph | Spring AI | Eino |
| Web 框架 | FastAPI | Spring Boot | Gin |
| 状态管理 | TypedDict + MemorySaver | POJO | struct |
| 向量检索 | FAISS | — | — |
| 缓存 | Redis | Redis | Redis |
| 追踪 | OpenTelemetry | OpenTelemetry | OpenTelemetry |
| 运行时 | Python 3.12 | Java 17+ | Go 1.22+ |

**版本一致性**：三套实现保持功能对等，方便横向对比编排框架的差异。

---

## 目录结构

```
.
├── docker-compose.yml           # 编排 Redis + python-agent + jaeger
├── docker-compose.override.yml  # 本地调试用：排除 jaeger / java / go
├── demo-ui.html                 # 零依赖演示页
│
├── python-impl/                 # Python 版（完整实现，评测覆盖）
│   ├── api/main.py              # FastAPI 入口与路由
│   ├── agents/
│   │   ├── supervisor.py        # 图编排、路由决策、结果汇总
│   │   ├── intent_router.py     # 意图分类与实体抽取
│   │   ├── knowledge_rag.py     # RAG 完整流程
│   │   ├── ticket_handler.py    # 工单处理
│   │   ├── compliance_checker.py# 两阶段合规审查
│   │   └── executor.py          # 节点超时控制与降级
│   ├── memory/
│   │   ├── working_memory.py    # 工作记忆（进程内）
│   │   ├── short_term.py        # 短期记忆（Redis）
│   │   ├── long_term.py         # 长期记忆（FAISS）+ 门禁
│   │   └── embedding.py         # 可插拔 embedding 实现
│   ├── mcp/mcp_server.py        # MCP 工具协议服务端
│   ├── tracing/otel_config.py   # OpenTelemetry 集成
│   ├── eval/                    # 检索质量评测
│   │   ├── eval_set.json        # 17 条标注 query
│   │   ├── embedding_backends.py# 评测用 embedding 对照实现
│   │   ├── metrics.py           # Recall@1 / MRR / 拒答率
│   │   └── run_eval.py          # CLI 入口
│   ├── tests/test_improvements.py
│   ├── docs/eval_report.md      # 评测报告
│   ├── requirements.txt
│   ├── Dockerfile
│   └── demo-ui.html
│
├── java-impl/                   # Java 版（Spring Boot + Spring AI）
│   ├── pom.xml
│   └── src/main/java/com/smartcs/
│
├── go-impl/                     # Go 版（Eino）
│   ├── go.mod
│   ├── main.go
│   └── agent/ memory/ mcp/ tracing/
│
└── docs/                        # 架构与部署文档
    ├── architecture.md
    ├── code-walkthrough.md
    └── deployment.md
```

---

## 安装与运行

### 前置条件

- Python 3.11+
- 一个 LLM API Key（支持 OpenAI 兼容协议，如 DeepSeek、OpenAI）
- 可选：Docker + Docker Compose

### 方式一：本地运行

```bash
cd python-impl

# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置
cp .env.example .env
# 编辑 .env，至少填写 OPENAI_API_KEY 与 OPENAI_BASE_URL

# 3. 启动
python -m api.main
```

访问：

| 地址 | 说明 |
|------|------|
| `http://localhost:8000/` | 聊天演示页 |
| `http://localhost:8000/docs` | Swagger 文档 |
| `http://localhost:8000/redoc` | ReDoc 文档 |

### 方式二：Docker Compose

```bash
# 1. 在项目根目录配置（compose 从根目录读 .env）
cp python-impl/.env.example python-impl/.env
# 编辑 python-impl/.env，填写 API Key

# 2. 启动
docker compose up -d
```

`docker-compose.override.yml` 已将 `jaeger` / `java-agent` / `go-agent` 设为可选 profile，默认只启动 `redis` 与 `python-agent`。

如需 Jaeger 追踪界面，编辑 override 文件去掉 jaeger 的 profile限制后：

```bash
docker compose --profile optional up -d
```

### 方式三：Java 版

```bash
cd java-impl
mvn clean package -DskipTests
java -jar target/smart-cs-agent-1.0.0.jar
```

### 方式四：Go 版

```bash
cd go-impl
go mod tidy
go run main.go
```

---

## 配置项

`python-impl/.env`：

| 变量 | 必填 | 默认值 | 说明 |
|------|------|--------|------|
| `OPENAI_API_KEY` | 是 | — | API 密钥 |
| `OPENAI_BASE_URL` | 是 | — | OpenAI 兼容端点，DeepSeek 为 `https://api.deepseek.com`（**不带 /v1**） |
| `MODEL_NAME` | 否 | `deepseek-flash` | 对话模型 |
| `EMBEDDING_TYPE` | 否 | `keyword` | embedding 实现：`keyword` / `openai` / `fake` |
| `EMBEDDING_MODEL` | 否 | `text-embedding-3-small` | `EMBEDDING_TYPE=openai` 时生效 |
| `REDIS_URL` | 否 | `redis://localhost:6379/0` | 短期记忆连接串 |
| `FAISS_INDEX_PATH` | 否 | `./vector_store/faiss_index` | 向量索引落盘路径 |
| `OTEL_SERVICE_NAME` | 否 | `smart-cs-multi-agent` | 追踪服务名 |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | 否 | 空 | OTLP 接收端，不设则输出到控制台 |
| `HOST` / `PORT` | 否 | `0.0.0.0` / `8000` | 监听地址 |

### 切换 embedding

```bash
# 使用字符二元组（默认，离线可用）
EMBEDDING_TYPE=keyword

# 使用 OpenAI 兼容 embedding
EMBEDDING_TYPE=openai
EMBEDDING_MODEL=text-embedding-3-small
```

---

## API 说明

### POST /api/chat

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `message` | string | 是 | 用户消息 |
| `user_id` | string | 否 | 用户标识，默认 `anonymous` |
| `session_id` | string | 否 | 会话 ID，不传则新建 |

响应：

```json
{
  "response": "**退款政策**\n\n购买后 7 天内可申请无理由退款……",
  "session_id": "99b9115d-ec17-4b01-bed7-e26e604aedc8",
  "intent": "knowledge_rag",
  "compliance_passed": true
}
```

示例：

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "怎么退款", "user_id": "user_001"}'
```

### 其他接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/health` | 健康检查 |
| GET | `/api/history/{session_id}` | 查询历史消息 |
| GET | `/api/tools` | 列出已注册工具 |
| POST | `/api/tools/call` | 调用指定工具 |
| GET | `/api/metrics` | Agent 耗时与工具调用指标 |

---

## 评测集使用

### 运行评测

```bash
cd python-impl

# 伪随机 embedding 基线
python -m eval.run_eval --backend fake

# 字符二元组（默认）
python -m eval.run_eval --backend keyword

# 启用相似度门禁
python -m eval.run_eval --backend keyword --threshold 0.01

# 逐条打印检索明细
python -m eval.run_eval --backend keyword --verbose

# JSON 输出，便于程序化对比
python -m eval.run_eval --backend keyword --threshold 0.01 --json
```

### 运行测试

```bash
python -m unittest discover -s tests -v
```

测试覆盖超时降级、合规超时拦截、异常隔离、结果聚合类型处理、意图归一化。

### 扩展评测集

编辑 `eval/eval_set.json`，每个 query 包含：

```json
{
  "query": "怎么退款",
  "expected_source": "refund_policy.md",
  "category": "in_domain"
}
```

`expected_source` 设为 `null` 即视为域外样本，用于统计拒答率。

---

## 常见问题

### 合规审查超时为什么不放行

超时按「不通过」处理是刻意的取舍。合规节点的错误代价不对称：放行的代价是一条违规回复可能带来监管处罚；拦截的代价只是用户多等待几分钟。在金融场景下，宁可误拦。

### 检索到了文档，但内容不相关怎么办

已通过相似度门禁处理。检索最高分低于阈值时直接返回空结果，上层返回固定话术并建议转人工，而不是拿低分文档生成答案。评测显示该策略在召回率零损失的前提下把域外拒答率从 0% 提升到 100%。

门限值需随语料规模重新标定，当前 0.01 基于 3 篇文档的评测集得出。

### 端口被占用怎么办

修改 `docker-compose.override.yml` 中的端口映射：

```yaml
services:
  python-agent:
    ports:
      - "8010:8000"
```

### Docker 拉取镜像失败

国内网络下 `jaegertracing/all-in-one` 可能拉取失败。`docker-compose.override.yml` 已默认排除该服务，核心功能不依赖它。

若其他镜像也失败，可配置镜像加速器，或手动拉取后改用本地镜像名。

### embedding 用字符二元组效果如何

在当前评测集上 Recall@1 为 100%，且完全离线。局限是它只能捕捉字面重合，对完全同义但用词不同的表达无能为力——这类场景需要 `openai` 后端的真实语义 embedding。

### 短期记忆为什么没有生效

`short_term.py` 已实现 Redis 存储与 `get_context_window()`，但编排图中没有节点调用该方法，因此只写不读。多轮上下文目前依赖 LangGraph 的 `MemorySaver` 检查点。接通这一层需要改造节点间的上下文传递方式。

### 模型返回的意图名不合法怎么办

Supervisor 的路由决策会做归一化处理：先剥离非字母字符，再做包含匹配，最后用白名单校验。不匹配时回退到 `knowledge_rag`。

---

## 当前限制

明确列出尚未完成的部分，避免误判项目完成度：

- **评测仅覆盖检索层。** 生成层的 Faithfulness（回答是否忠实于文档）与 Answer Relevance 未测，前者需要 LLM-as-Judge
- **端到端指标缺失。** FCR、CSAT 需要真实流量积累
- **短期记忆未接通。** 只写不读，多轮依赖检查点
- **混合检索未实现。** 目前仅字符二元组一路，BM25 + 向量融合（RRF）待做
- **Java 版与 Go 版未验证。** 仅确认代码完整，未构建运行
- **重排依赖 LLM。** `knowledge_rag.py` 用 LLM 评估相关性重排，延迟较高，未接入专用 rerank 模型
- **门限基于小语料。** 0.01 由 3 篇文档得出，语料扩大后需重新标定

---

## 许可证

[MIT](./LICENSE)

原项目作者与版权声明保留。
