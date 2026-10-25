"""
Boolean Model (IR Module 2)
---------------------------
A Boolean query is a logic expression over index terms, e.g.

    paneer AND (butter OR cream) NOT onion

Each term is replaced by the SET of documents that contain it (taken
straight from the inverted index), and the operators become set operations:

    AND -> intersection      OR -> union      NOT -> complement (all docs − set)

The result is a SET of matching documents — the Boolean model cannot rank
them or express partial matches. That is exactly why we use it for hard
constraints (diet, cuisine, excluded ingredients) and hand its result to a
ranking model (VSM / BM25) as the candidate set.

Parsing: the expression is converted to postfix (Reverse Polish) with the
shunting-yard algorithm, then evaluated with a stack.
"""

import re

from ir.inverted_index import InvertedIndex
from ir.preprocess import preprocess

_PRECEDENCE = {"NOT": 3, "AND": 2, "OR": 1}
_TOKEN_RE = re.compile(r"\(|\)|\bAND\b|\bOR\b|\bNOT\b|[A-Za-z]+", re.IGNORECASE)


class BooleanModel:
    def __init__(self, index: InvertedIndex):
        self.index = index
        self.universe: set[int] = set(index.doc_len.keys())

    # ---------------------------------------------------------------- parsing
    def tokenize(self, expression: str) -> list[str]:
        """Split into terms, operators and parentheses; clean the terms with
        the same text operations as the documents; insert implicit ANDs."""
        raw = _TOKEN_RE.findall(expression)
        tokens: list[str] = []
        for tok in raw:
            up = tok.upper()
            if up in _PRECEDENCE or tok in "()":
                tokens.append(up if up in _PRECEDENCE else tok)
            else:
                cleaned = preprocess(tok)
                if cleaned:                      # stop words vanish
                    tokens.append(cleaned[0])
        # implicit AND between two adjacent operands, e.g. "paneer butter"
        out: list[str] = []
        for tok in tokens:
            if out and self._is_operand_end(out[-1]) and self._is_operand_start(tok):
                out.append("AND")
            out.append(tok)
        return out

    @staticmethod
    def _is_operand_end(tok: str) -> bool:
        return tok == ")" or (tok not in _PRECEDENCE and tok != "(")

    @staticmethod
    def _is_operand_start(tok: str) -> bool:
        return tok == "(" or tok == "NOT" or (tok not in _PRECEDENCE and tok != ")")

    def to_postfix(self, tokens: list[str]) -> list[str]:
        """Shunting-yard: infix -> postfix. NOT binds tightest, then AND, then OR."""
        output: list[str] = []
        stack: list[str] = []
        for tok in tokens:
            if tok == "(":
                stack.append(tok)
            elif tok == ")":
                while stack and stack[-1] != "(":
                    output.append(stack.pop())
                if stack:
                    stack.pop()                  # discard "("
            elif tok in _PRECEDENCE:
                while (stack and stack[-1] in _PRECEDENCE
                       and _PRECEDENCE[stack[-1]] >= _PRECEDENCE[tok] and tok != "NOT"):
                    output.append(stack.pop())
                stack.append(tok)
            else:
                output.append(tok)
        while stack:
            op = stack.pop()
            if op != "(":
                output.append(op)
        return output

    # ---------------------------------------------------------------- evaluation
    def evaluate(self, expression: str) -> tuple[set[int], list[str]]:
        """Return (matching doc_ids, postfix tokens used)."""
        postfix = self.to_postfix(self.tokenize(expression))
        stack: list[set[int]] = []
        for tok in postfix:
            if tok == "NOT":
                stack.append(self.universe - (stack.pop() if stack else set()))
            elif tok in ("AND", "OR"):
                b = stack.pop() if stack else set()
                a = stack.pop() if stack else set()
                stack.append(a & b if tok == "AND" else a | b)
            else:
                stack.append(set(self.index.get_postings(tok).keys()))
        return (stack.pop() if stack else set()), postfix
