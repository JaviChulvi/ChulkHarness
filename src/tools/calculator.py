"""Safe arithmetic calculator tool."""

from __future__ import annotations

import ast
import operator
from typing import Any

from src.tools.registry import Tool, ToolResult


_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}
_UNARY_OPERATORS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def calculate(arguments: dict[str, Any]) -> ToolResult:
    """Evaluate a simple arithmetic expression."""
    expression = arguments["expression"]
    try:
        value = _safe_eval(expression)
    except Exception as exc:
        return ToolResult(
            tool_name="calculator",
            success=False,
            observation=f"Invalid arithmetic expression: {exc}",
            error="invalid_expression",
        )
    return ToolResult(tool_name="calculator", success=True, observation=f"{expression} = {value}")


def calculator_tool() -> Tool:
    """Create the calculator tool definition."""
    return Tool(
        name="calculator",
        description="Evaluate simple arithmetic expressions with +, -, *, /, %, **, and parentheses.",
        args_schema={
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Arithmetic expression to evaluate, for example: (2 + 3) * 4",
                    "minLength": 1,
                    "maxLength": 200,
                }
            },
            "required": ["expression"],
            "additionalProperties": False,
        },
        callable=calculate,
    )


def _safe_eval(expression: str) -> int | float:
    if len(expression) > 200:
        raise ValueError("expression is too long")
    tree = ast.parse(expression, mode="eval")
    return _eval_node(tree.body)


def _eval_node(node: ast.AST) -> int | float:
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 10:
            raise ValueError("exponent is too large")
        return _BINARY_OPERATORS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
        return _UNARY_OPERATORS[type(node.op)](_eval_node(node.operand))
    raise ValueError(f"unsupported expression element: {type(node).__name__}")
