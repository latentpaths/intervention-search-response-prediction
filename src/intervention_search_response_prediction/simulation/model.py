"""Small, deterministic simulator for the Boolean rules in boolmore's .bnet files."""

from __future__ import annotations

import ast
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

Action = tuple[tuple[int, int], ...]
_TOKEN = re.compile(r"[A-Za-z_][A-Za-z_0-9]*")


def _parse_expression(source: str) -> ast.expr:
    text = source.replace("!", " not ").replace("&", " and ").replace("|", " or ")
    tree = ast.parse(text.strip(), mode="eval").body

    def validate(node: ast.AST) -> None:
        if isinstance(node, ast.Name):
            return
        if isinstance(node, ast.Constant) and node.value in (0, 1):
            return
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            validate(node.operand)
            return
        if isinstance(node, ast.BoolOp) and isinstance(node.op, (ast.And, ast.Or)):
            for value in node.values:
                validate(value)
            return
        raise ValueError(f"unsupported Boolean rule: {source!r}")

    validate(tree)
    return tree


def _evaluate(node: ast.expr, values: dict[str, bool]) -> bool:
    if isinstance(node, ast.Name):
        return values[node.id]
    if isinstance(node, ast.Constant):
        return bool(node.value)
    if isinstance(node, ast.UnaryOp):
        return not _evaluate(node.operand, values)
    if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.And):
        return all(_evaluate(child, values) for child in node.values)
    if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.Or):
        return any(_evaluate(child, values) for child in node.values)
    raise AssertionError("expression was validated")


@dataclass(frozen=True)
class Rule:
    dependencies: tuple[int, ...]
    table: tuple[int, ...]

    def value(self, state: int) -> int:
        row = sum(((state >> node) & 1) << bit for bit, node in enumerate(self.dependencies))
        return self.table[row]

    def signs(self) -> tuple[int, ...]:
        """+1 activation, -1 inhibition, 0 mixed or ineffective."""
        result = []
        for bit in range(len(self.dependencies)):
            up = down = False
            for row in range(len(self.table)):
                if row & (1 << bit):
                    continue
                low, high = self.table[row], self.table[row | (1 << bit)]
                up |= high > low
                down |= high < low
            result.append(1 if up and not down else -1 if down and not up else 0)
        return tuple(result)


@dataclass(frozen=True)
class BooleanModel:
    name: str
    nodes: tuple[str, ...]
    rules: tuple[Rule, ...]
    sha256: str

    @classmethod
    def from_bnet(cls, path: Path) -> BooleanModel:
        raw = path.read_bytes()
        rows = []
        for line in raw.decode("utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or line.lower().startswith("targets,"):
                continue
            if "," not in line:
                raise ValueError(f"invalid .bnet row in {path}: {line}")
            node, expression = line.split(",", 1)
            rows.append((node.strip(), expression.strip()))
        names = tuple(node for node, _ in rows)
        if len(names) != len(set(names)):
            raise ValueError(f"duplicate node in {path}")
        name_to_index = {node: i for i, node in enumerate(names)}
        rules = []
        for node, expression in rows:
            identifiers = set(_TOKEN.findall(expression))
            unknown = identifiers - name_to_index.keys()
            if unknown:
                raise ValueError(f"undefined regulators for {node}: {sorted(unknown)}")
            dependencies = tuple(sorted(identifiers, key=name_to_index.__getitem__))
            syntax = _parse_expression(expression)
            table = []
            for row in range(1 << len(dependencies)):
                values = {name: bool((row >> bit) & 1) for bit, name in enumerate(dependencies)}
                table.append(int(_evaluate(syntax, values)))
            rules.append(Rule(tuple(name_to_index[name] for name in dependencies), tuple(table)))
        return cls(path.stem, names, tuple(rules), hashlib.sha256(raw).hexdigest())

    def signed_edges(self) -> tuple[tuple[int, int, int], ...]:
        return tuple(
            (source, target, sign)
            for target, rule in enumerate(self.rules)
            for source, sign in zip(rule.dependencies, rule.signs())
            if sign
        )

    def step(self, state: int, action: Action = ()) -> int:
        next_state = sum(rule.value(state) << i for i, rule in enumerate(self.rules))
        for node, value in action:
            if value:
                next_state |= 1 << node
            else:
                next_state &= ~(1 << node)
        return next_state

    def simulate(self, initial: int, action: Action, horizon: int) -> int:
        if horizon < 1:
            raise ValueError("horizon must be positive")
        state = initial
        for node, value in action:
            if value:
                state |= 1 << node
            else:
                state &= ~(1 << node)
        for _ in range(horizon):
            state = self.step(state, action)
        return state


def legal_actions(nodes: int, forbidden: frozenset[int] = frozenset()) -> tuple[Action, ...]:
    """All nonempty one- or two-node clamps, in stable order."""
    allowed = [node for node in range(nodes) if node not in forbidden]
    single = [((node, value),) for node in allowed for value in (0, 1)]
    double = [
        ((a, av), (b, bv))
        for i, a in enumerate(allowed)
        for b in allowed[i + 1 :]
        for av in (0, 1)
        for bv in (0, 1)
    ]
    return tuple(single + double)
