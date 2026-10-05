"""
RAG 检索质量评测脚本。

用途：量化检索层的真实表现，替代简历上编造的数字。
当前基线用的是项目自带的_sha256 伪随机 embedding（必然很差），
本脚本的作用是把"差"变成可量化的数字，作为后续优化的对照。

用法：
    python -m eval.run_eval            # 默认基线
    python -m eval.run_eval --json     # 输出 JSON，便于程序化对比
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# 允许直接 python eval/run_eval.py 运行
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.metrics import EvalReport, evaluate_retrieval  # noqa: E402
from eval.embedding_backends import EMBEDDING_BACKENDS  # noqa: E402

EVAL_FILE = Path(__file__).resolve().parent / "eval_set.json"


def load_cases() -> dict:
    return json.loads(EVAL_FILE.read_text(encoding="utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--backend",
        default="baseline",
        choices=list(EMBEDDING_BACKENDS.keys()),
        help="embedding 实现：baseline=项目自带伪随机；其余为真实实现",
    )
    ap.add_argument("--json", action="store_true", help="输出 JSON")
    ap.add_argument("--verbose", action="store_true", help="打印每条 query 的检索明细")
    ap.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="相似度门限，低于该值判为知识库外并拒答。默认 0 表示不启用",
    )
    args = ap.parse_args()

    data = load_cases()
    report: EvalReport = evaluate_retrieval(
        queries=data["queries"],
        backend_name=args.backend,
        verbose=args.verbose,
        threshold=args.threshold,
    )

    if args.json:
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    else:
        print(report.render())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
