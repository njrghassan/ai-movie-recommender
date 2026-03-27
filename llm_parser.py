from __future__ import annotations

import json
import os
import re
import functools
import urllib.error
import urllib.request
from typing import Any

from nl_query import GENRES, parse_query


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

@functools.lru_cache(maxsize=1)
def _dotenv_fallback() -> dict[str, str]:
    env: dict[str, str] = {}
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#") or "=" not in s:
                    continue
                k, v = s.split("=", 1)
                k = k.strip()
                v = v.strip().strip("'").strip('"')
                if k and v:
                    env[k] = v
    except Exception:
        return {}
    return env


def _get_env(name: str) -> str:
    v = os.getenv(name, "").strip()
    if v:
        return v
    return (_dotenv_fallback().get(name) or "").strip()


def _extract_first_json_object(text: str) -> dict[str, Any] | None:
    """
    Best-effort extraction of the first JSON object from model output.
    Handles common wrappers like ```json ... ``` or extra prose.
    """
    if not isinstance(text, str):
        return None

    cleaned = text.strip()
    if not cleaned:
        return None

    # Strip fenced code blocks if present.
    cleaned = re.sub(r"^\s*```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```\s*$", "", cleaned)

    # Fast path: whole response is JSON.
    try:
        obj = json.loads(cleaned)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    # Otherwise scan for the first balanced {...} region.
    start = cleaned.find("{")
    if start == -1:
        return None

    depth = 0
    for i in range(start, len(cleaned)):
        ch = cleaned[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                snippet = cleaned[start : i + 1]
                try:
                    obj = json.loads(snippet)
                    return obj if isinstance(obj, dict) else None
                except Exception:
                    return None
    return None


def _to_int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value != value:  # NaN
            return None
        return int(value)
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        try:
            return int(float(s))
        except Exception:
            return None
    return None


def _to_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        if isinstance(value, float) and value != value:  # NaN
            return None
        return float(value)
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        try:
            return float(s)
        except Exception:
            return None
    return None


def _normalize_genres(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    allowed = {g.lower(): g for g in GENRES}
    out: list[str] = []
    for item in value:
        if not isinstance(item, str):
            continue
        g = item.strip().lower()
        if not g:
            continue
        if g in allowed:
            out.append(allowed[g])
    # de-dupe preserving order
    seen: set[str] = set()
    deduped: list[str] = []
    for g in out:
        if g not in seen:
            seen.add(g)
            deduped.append(g)
    return deduped


def parse_query_llm(query: str) -> dict:
    """
    Convert a natural-language query into a structured filter dictionary using OpenRouter.

    Fallback behavior:
      - If OpenRouter is unavailable (missing key / network / API error / invalid JSON),
        falls back to the existing rule-based `nl_query.parse_query()`.
    """
    q = (query or "").strip()
    if not q:
        return parse_query(q)

    api_key = _get_env("OPENROUTER_API_KEY")
    if not api_key:
        return parse_query(q)

    model = _get_env("OPENROUTER_MODEL") or "mistralai/mistral-7b-instruct"

    system_prompt = (
        "You convert user movie search queries into a JSON object of filters.\n"
        "Return ONLY valid JSON (no markdown, no code fences, no explanation).\n"
        "Use null when a field is not specified.\n"
        "Genres must be chosen only from this list:\n"
        f"{GENRES}\n"
        "Schema:\n"
        '{\n'
        '  "min_year": number|null,\n'
        '  "max_year": number|null,\n'
        '  "genres": string[],\n'
        '  "person": string|null,\n'
        '  "min_rating": number|null\n'
        "}\n"
        'If the query contains no usable constraints, return all null/empty values.'
    )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    site_url = _get_env("OPENROUTER_SITE_URL")
    app_name = _get_env("OPENROUTER_APP_NAME")
    if site_url:
        headers["HTTP-Referer"] = site_url
    if app_name:
        headers["X-Title"] = app_name

    payload = {
        "model": model,
        "temperature": 0.1,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": q},
        ],
    }

    try:
        req = urllib.request.Request(
            OPENROUTER_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=20) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        data = json.loads(raw)
        content = (
            (data.get("choices") or [{}])[0]
            .get("message", {})
            .get("content", "")
        )
        obj = _extract_first_json_object(content)
        if not isinstance(obj, dict):
            return parse_query(q)

        filters: dict[str, Any] = {
            "min_year": _to_int_or_none(obj.get("min_year")),
            "max_year": _to_int_or_none(obj.get("max_year")),
            "genres": _normalize_genres(obj.get("genres")),
            "person": obj.get("person").strip() if isinstance(obj.get("person"), str) and obj.get("person").strip() else None,
            "min_rating": _to_float_or_none(obj.get("min_rating")),
            # Keep compatibility with existing filter pipeline defaults.
            "genre_mode": "OR",
        }
        return filters
    except (urllib.error.URLError, urllib.error.HTTPError, ValueError, Exception):
        return parse_query(q)
