"""
检索质量评估指标。

关键指标定义（面试可讲清出处）：
- Recall@1：正确答案排在第一位的比例。反映"检索准不准"。
- MRR：所有域内 query 的倒数排名均值。反映"正确答案排得靠不靠前"。
- OOD 拒答率：知识库外的问题，系统是否能识别出"不该回答"，
  而不是硬答。这是 RAG 抗幻觉的核心指标。

区分域内 / 域外是本评测集的设计要点：
只测域内会掩盖幻觉问题（项目原模板正是这么自我美化的）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from eval.embedding_backends import get_backend


@dataclass
class QueryResult:
    query: str
    category: str
    expected_source: str | None
    retrieved_sources: list[str]
    top1_correct: bool
    reciprocal_rank: float
    # 域外问题：系统是否有依据拒答
    ood: bool = False
    ood_answered: bool = False
    top_score: float = 0.0


@dataclass
class EvalReport:
    total: int = 0
    in_domain: int = 0
    out_of_domain: int = 0
    top1_hits: int = 0
    reciprocal_rank_sum: float = 0.0
    ood_correctly_refused: int = 0
    per_category: dict[str, dict[str, int]] = field(default_factory=dict)
    details: list[QueryResult] = field(default_factory=list)

    @property
    def recall_at_1(self) -> float:
        """域内 Recall@1"""
        return self.top1_hits / self.in_domain if self.in_domain else 0.0

    @property
    def mrr(self) -> float:
        """域内 MRR"""
        return self.reciprocal_rank_sum / self.in_domain if self.in_domain else 0.0

    @property
    def ood_refusal_rate(self) -> float:
        """域外拒答率：越高说明越不容易瞎答"""
        return self.ood_correctly_refused / self.out_of_domain if self.out_of_domain else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "in_domain": self.in_domain,
            "out_of_domain": self.out_of_domain,
            "metrics": {
                "recall_at_1": round(self.recall_at_1, 4),
                "mrr": round(self.mrr, 4),
                "ood_refusal_rate": round(self.ood_refusal_rate, 4),
            },
            "per_category": self.per_category,
            "details": [
                {
                    "query": d.query,
                    "category": d.category,
                    "expected": d.expected_source,
                    "retrieved": d.retrieved_sources,
                    "top1_correct": d.top1_correct,
                }
                for d in self.details
            ],
        }

    def render(self) -> str:
        lines = [
            "=" * 62,
            "RAG 检索质量评测报告",
            "=" * 62,
            f"评测样本     : {self.total} 条（域内 {self.in_domain} / 域外 {self.out_of_domain}）",
            "",
            f"Recall@1     : {self.recall_at_1 * 100:5.1f}%   正确答案排第一位的比例",
            f"MRR          : {self.mrr:.3f}      正确答案排名的倒数均值",
            f"OOD 拒答率   : {self.ood_refusal_rate * 100:5.1f}%   域外问题不瞎答的比例",
            "",
            "分类明细:",
        ]
        for cat, st in sorted(self.per_category.items()):
            total = st["total"]
            hit = st["top1_hits"]
            rate = hit / total * 100 if total else 0.0
            lines.append(f"  {cat:24s} {hit}/{total}  ({rate:5.1f}%)")
        lines += [
            "",
            "-" * 62,
            "逐条明细:",
        ]
        for d in self.details:
            mark = "OK " if d.top1_correct else "MISS"
            if d.ood:
                mark = "拒答" if not d.ood_answered else "瞎答"
            exp = d.expected_source or "(域外)"
            got = ", ".join(d.retrieved_sources[:3]) or "(空)"
            lines.append(f"[{mark}] {d.query}")
            lines.append(f"        期望: {exp}")
            lines.append(f"        实得: {got}")
        lines.append("=" * 62)
        return "\n".join(lines)


def evaluate_retrieval(
    queries: list[dict],
    backend_name: str = "baseline",
    verbose: bool = False,
    threshold: float | None = None,
) -> EvalReport:
    """
    在指定 embedding backend 上跑完整评测。

    Args:
        threshold: 相似度门限。启用后，域外问题若最高分仍低于门限，
                   视为"正确拒答"；域内问题若低于门限则计为漏答。
                   这模拟线上的拒答门禁策略。
    """
    backend = get_backend(backend_name)
    report = EvalReport()

    for item in queries:
        query = item["query"]
        expected = item.get("expected_source")
        category = item.get("category", "unknown")
        is_ood = expected is None

        hits = backend.search(query, top_k=3)
        retrieved = [h["source"] for h in hits]
        scores = [round(h["score"], 4) for h in hits]
        top_score = scores[0] if scores else 0.0

        if verbose:
            print(f"  {query} -> {retrieved} scores={scores}")

        # 应用拒答门禁
        refused = threshold is not None and top_score < threshold
        effective_retrieved = [] if refused else retrieved

        # 计算 Reciprocal Rank
        rr = 0.0
        top1_correct = False
        for idx, src in enumerate(effective_retrieved):
            if src == expected:
                rr = 1.0 / (idx + 1)
                top1_correct = idx == 0
                break

        if is_ood:
            report.out_of_domain += 1
            if refused:
                report.ood_correctly_refused += 1
        else:
            report.in_domain += 1
            if top1_correct:
                report.top1_hits += 1
                report.reciprocal_rank_sum += rr

        report.total += 1
        cat_stat = report.per_category.setdefault(
            category, {"total": 0, "top1_hits": 0}
        )
        cat_stat["total"] += 1
        if top1_correct:
            cat_stat["top1_hits"] += 1

        report.details.append(
            QueryResult(
                query=query,
                category=category,
                expected_source=expected,
                retrieved_sources=retrieved,
                top1_correct=top1_correct,
                reciprocal_rank=rr,
                ood=is_ood,
                ood_answered=not refused,
                top_score=top_score,
            )
        )

    return report
