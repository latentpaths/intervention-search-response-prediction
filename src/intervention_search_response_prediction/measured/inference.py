"""Bounded, auditable Prime requests with a shared persistent spending ledger."""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import random
import threading
import time
from pathlib import Path

from openai import OpenAI

from ..paths import INFERENCE_LEDGER, MEASURED
from .environment import MeasuredEpisode
from .ledger import reconcile
from .metrics import result
from .procedures import Ridge
from .protocol import PROTOCOL

SYSTEM = (
    "You predict measured biological perturbation responses in retrospective screen replay. "
    "Only disclosed single responses and observed assay feedback are available. "
    "Return one JSON object only. Predictions are continuous real numbers in the specified units, not probabilities. "
    "Follow array order exactly. Never invent an assay result. Keep every note within 600 characters. "
)
NOTES = {
    "summary": "In note, briefly summarize observations. Do not maintain a mechanistic hypothesis.",
    "hypothesis": "In note, maintain a short response hypothesis and revise it when observed responses contradict predictions.",
}


class BudgetExceeded(RuntimeError):
    pass


class Calls:
    def __init__(self, directory=None, client=None):
        from ..paths import repository_root, writable_output

        directory = writable_output(
            directory if directory is not None else repository_root() / f"outputs/{MEASURED}"
        )
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.file = self.directory / INFERENCE_LEDGER
        self.lock = threading.RLock()
        # Refuse concurrent runner processes; threads share one ledger.
        self.process_lock = (self.directory / "inference.lock").open("a")
        fcntl.flock(self.process_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.events = (
            [json.loads(x) for x in self.file.read_text().splitlines()]
            if self.file.exists()
            else []
        )
        self.cache = {x["call_id"]: x for x in self.events if x["event"] == "response"}
        self.spent, _ = reconcile(self.events)
        self.reservations = {}
        if client is None:
            key = os.environ.get("PRIME_API_KEY")
            p = Path.home() / ".prime/config.json"
            if not key and p.exists():
                key = json.loads(p.read_text()).get("api_key")
            if not key:
                raise RuntimeError("Prime authentication unavailable")
            client = OpenAI(
                api_key=key, base_url="https://api.pinference.ai/api/v1", timeout=90, max_retries=0
            )
        self.client = client

    def append(self, obj):
        with self.file.open("a") as f:
            f.write(json.dumps(obj, sort_keys=True, allow_nan=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        self.events.append(obj)

    def reserve_block(self, block, phase, amount):
        with self.lock:
            limit = PROTOCOL[phase + "_cap_usd"]
            reserved = sum(v[1] for v in self.reservations.values() if v[0] == phase)
            if (
                self.spent[phase] + reserved + amount > limit
                or sum(self.spent.values()) + sum(v[1] for v in self.reservations.values()) + amount
                > 50
            ):
                raise BudgetExceeded(
                    f"{phase} cannot reserve complete block {block}; spent={self.spent}"
                )
            self.reservations[block] = [phase, amount]

    def release_block(self, block):
        with self.lock:
            self.reservations.pop(block, None)

    def ask(self, call_id, system, message, phase, block):
        request = {
            "model": PROTOCOL["model"],
            "system": system,
            "message": message,
            "max_tokens": 1536,
            "temperature": 0,
            "reasoning_effort": "none",
        }
        h = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
        with self.lock:
            if call_id in self.cache:
                r = self.cache[call_id]
                if r["request_hash"] != h:
                    raise ValueError("cached request mismatch")
                return {**r, "cache_hit": True}
            # UTF-8 byte count is a conservative token bound, plus message overhead.
            reserve = 2e-6 * (len(system.encode()) + len(message.encode()) + 512) + 10e-6 * 1536
            if block not in self.reservations or reserve > self.reservations[block][1]:
                raise BudgetExceeded("block reservation exhausted")
            self.reservations[block][1] -= reserve
            self.spent[phase] += reserve
            self.append(
                {
                    "event": "reserve",
                    "call_id": call_id,
                    "phase": phase,
                    "request_hash": h,
                    "reserve_usd": reserve,
                    "timestamp": time.time(),
                }
            )
        tick = time.perf_counter()
        try:
            response = self.client.chat.completions.create(
                model=PROTOCOL["model"],
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": message},
                ],
                temperature=0,
                reasoning_effort="none",
                max_tokens=1536,
                response_format={"type": "json_object"},
                extra_body={"usage": {"include": True}},
            ).model_dump()
            usage = response.get("usage") or {}
            actual = usage.get("cost")
            if actual is None and "prompt_tokens" in usage and "completion_tokens" in usage:
                actual = usage["prompt_tokens"] * 2e-6 + usage["completion_tokens"] * 10e-6
            charged = reserve if actual is None else float(actual)
            if not math.isfinite(charged) or charged < 0:
                charged = reserve
            row = {
                "event": "response",
                "call_id": call_id,
                "phase": phase,
                "request_hash": h,
                "request": request,
                "model": response.get("model"),
                "usage": usage,
                "charged_usd": charged,
                "latency_seconds": time.perf_counter() - tick,
                "response_id": response.get("id"),
                "finish_reason": response["choices"][0].get("finish_reason"),
                "content": response["choices"][0]["message"].get("content") or "",
                "timestamp": time.time(),
            }
        except Exception as ex:
            with self.lock:
                self.append(
                    {
                        "event": "error",
                        "call_id": call_id,
                        "phase": phase,
                        "charged_usd": reserve,
                        "error_type": type(ex).__name__,
                        "timestamp": time.time(),
                    }
                )
            raise
        with self.lock:
            self.spent[phase] += charged - reserve
            self.reservations[block][1] += max(0, reserve - charged)
            self.append(row)
            self.cache[call_id] = row
            if charged > reserve:
                raise BudgetExceeded("provider charge exceeded conservative bound; stop")
        return {**row, "cache_hit": False}


def compact(value):
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, list):
        return [compact(x) for x in value]
    if isinstance(value, dict):
        return {k: compact(v) for k, v in value.items()}
    return value


def vector(value, width):
    return (
        isinstance(value, list)
        and len(value) == width
        and all(type(x) in (float, int) and math.isfinite(x) and abs(x) <= 1e6 for x in value)
    )


def valid(obj, kind, width, legal, n):
    if not isinstance(obj, dict):
        return False
    if kind == "probe":
        return (
            isinstance(obj.get("predictions"), list)
            and len(obj["predictions"]) == n
            and all(vector(p, width) for p in obj["predictions"])
        )
    if kind == "final":
        return isinstance(obj.get("action"), str) and obj["action"] in legal
    return (
        isinstance(obj.get("action"), str)
        and obj["action"] in legal
        and vector(obj.get("prediction"), width)
        and isinstance(obj.get("note"), str)
        and len(obj["note"]) <= 600
    )


def block_bound(t):
    # 22 calls per paired block, each with at most one repair; 12KB overhead
    # accommodates eight compact observation vectors, notes and probe IDs.
    size = len(json.dumps(compact(t.view()), separators=(",", ":")).encode()) + 12000
    return 44 * (2e-6 * (size + 2048) + 10e-6 * 1536)


def run_llm(t, condition, mode, calls, phase, block):
    begin = time.perf_counter()
    episode = MeasuredEpisode(t)
    view = compact(t.view())
    history = []
    note = ""
    call_rows = []
    failures = []
    timings = []
    prefix = f"{phase}:{t.id}:{mode}:{condition}"
    fixed = t.fixed_actions()
    shuffled = t.values(fixed).tolist()
    random.Random(f"shuffle:{t.id}").shuffle(shuffled)
    ridge = Ridge(t)

    def ask(kind, ids, step):
        nonlocal note
        message = {"phase": kind, "view": view, "history": history, "previous_note": note}
        if kind == "probe":
            message.update(
                actions=list(ids), format={"predictions": [[0.0] * len(t.readouts)] * len(ids)}
            )
        elif kind == "final":
            message.update(legal_actions=list(ids), format={"action": ids[0]})
        else:
            message.update(
                legal_actions=list(ids),
                format={
                    "action": ids[0],
                    "prediction": [0.0] * len(t.readouts),
                    "note": "brief note",
                },
            )
        base = json.dumps(compact(message), separators=(",", ":"))
        system = SYSTEM + NOTES[condition]
        for attempt in range(2):
            msg = (
                base
                if attempt == 0
                else base
                + "\nYour preceding response failed schema validation. Return exactly the requested keys, shapes and legal IDs. Do not add commentary."
            )
            row = calls.ask(f"{prefix}:{step}:try{attempt}", system, msg, phase, block)
            call_rows.append(row)
            try:
                obj = json.loads(row["content"])
            except (ValueError, TypeError):
                obj = None
            if valid(obj, kind, len(t.readouts), ids, len(ids)):
                if kind == "assay":
                    note = obj["note"]
                return obj
            failures.append(
                {
                    "step": step,
                    "attempt": attempt,
                    "call_id": row["call_id"],
                    "finish_reason": row["finish_reason"],
                }
            )
        # Identical deterministic fallback based only on the visible history.
        if kind == "probe":
            return {"predictions": ridge.predict(ids, history)[0].tolist()}
        if kind == "final":
            return {"action": max(history, key=lambda r: r["value"][0])["action"]}
        return {
            "action": ids[0],
            "prediction": ridge.predict([ids[0]], history)[0][0].tolist(),
            "note": note,
        }

    before = ask("probe", t.probes, "before")["predictions"]
    for step in range(8):
        legal = (
            tuple(a for a in t.candidates if a not in [h["action"] for h in history])
            if mode == "adaptive"
            else (fixed[step],)
        )
        tick = time.perf_counter()
        a = ask("assay", legal, f"assay{step}")
        episode.predict(a["action"], a["prediction"])
        timings.append(time.perf_counter() - tick)
        observation = episode.experiment(a["action"])
        # Prediction-free shared observations keep the fixed-history comparison matched.
        history.append(
            {
                "action": a["action"],
                "value": shuffled[step] if mode == "shuffled" else observation["value"],
            }
        )
    after = ask("probe", t.probes, "after")["predictions"]
    final = (
        ask("final", t.candidates, "final")["action"] if t.search else episode.history[-1]["action"]
    )
    episode.submit(final)
    return result(
        t,
        condition,
        mode,
        before,
        after,
        episode.history,
        final,
        timings,
        time.perf_counter() - begin,
        visible_history=history,
        failures=failures,
        call_ids=[r["call_id"] for r in call_rows],
        charged_usd=sum(r["charged_usd"] for r in call_rows),
        cache_hits=sum(r["cache_hit"] for r in call_rows),
        uncached_call_seconds=[r["latency_seconds"] for r in call_rows if not r["cache_hit"]],
        original_call_seconds=[r["latency_seconds"] for r in call_rows],
    )
