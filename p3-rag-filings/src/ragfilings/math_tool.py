"""Python Financial Math Tool.

Executes exact Python mathematical calculations (growth rates, deltas, percentages, CAGRs)
using safe AST evaluation to eliminate LLM arithmetic errors on 10-K financial tables.
"""

from __future__ import annotations

import ast
import json
import operator
from typing import Any

from . import generation, verification

_SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def safe_eval(expr: str) -> float:
    """Safely evaluate a mathematical Python expression using AST parsing."""
    def _eval_node(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        elif isinstance(node, ast.BinOp):
            op = _SAFE_OPERATORS.get(type(node.op))
            if op is None:
                raise ValueError(f"Unsupported operator: {type(node.op)}")
            return op(_eval_node(node.left), _eval_node(node.right))
        elif isinstance(node, ast.UnaryOp):
            op = _SAFE_OPERATORS.get(type(node.op))
            if op is None:
                raise ValueError(f"Unsupported unary operator: {type(node.op)}")
            return op(_eval_node(node.operand))
        else:
            raise ValueError(f"Unsupported AST node: {type(node)}")

    parsed = ast.parse(expr, mode='eval')
    return _eval_node(parsed.body)


def compute_financial_math(query: str, context_chunks: list[dict[str, Any]], cfg: dict) -> dict[str, Any] | None:
    """Extract figures, generate Python math expression, and execute it safely."""
    context_text = "\n".join(c.get("text", "") for c in context_chunks[:6])
    messages = [
        {
            "role": "system",
            "content": (
                "You are a financial calculation assistant. Given retrieved 10-K text and a question, "
                "extract the exact figures and output a single Python mathematical expression to calculate the answer. "
                "Output ONLY a JSON object: {\"expression\": \"<python_math_expression>\", \"explanation\": \"<short note>\"}. "
                "Example: {\"expression\": \"(416161 - 383285) / 383285 * 100\", \"explanation\": \"Growth rate from 2023 to 2025\"}"
            )
        },
        {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {query}"}
    ]

    try:
        text, _ = generation._complete(messages, cfg)
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            data = json.loads(text[start:end+1])
            expr = data.get("expression", "")
            if expr:
                val = safe_eval(expr)
                return {
                    "expression": expr,
                    "result_value": round(val, 4),
                    "formatted": f"{val:.2f}%" if "growth" in query.lower() or "percent" in query.lower() or "margin" in query.lower() else f"{val:,.2f}",
                    "explanation": data.get("explanation", "")
                }
    except Exception:
        pass

    return None
