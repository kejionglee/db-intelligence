from __future__ import annotations

import time
from typing import List, Optional, Tuple, Dict

import requests

from .config import SETTINGS


_CHAT_CACHE: Dict[Tuple[str, str, float], Tuple[float, str]] = {}
_EMBED_CACHE: Dict[str, Tuple[float, List[float]]] = {}
_CACHE_TTL_SECONDS = 300


class LlmClient:
    def __init__(self, model: Optional[str] = None) -> None:
        # Default to chat model for chats
        self._model = model or SETTINGS.ollama_model_chat or SETTINGS.ollama_model
        self._ollama_host = SETTINGS.ollama_host

    @property
    def model(self) -> str:
        return self._model

    def chat(self, system: str, user: str, temperature: float = 0.0, timeout: Optional[float] = None) -> str:
        key = (system, user, float(temperature))
        now = time.time()
        cached = _CHAT_CACHE.get(key)
        if cached and now - cached[0] < _CACHE_TTL_SECONDS:
            return cached[1]
        try:
            result = self._chat_ollama(system=system, user=user, temperature=temperature, timeout=timeout)
        except Exception:
            raise
        # Strip <think> tags if present
        result = result.replace("<think>", "").replace("</think>", "").strip()
        _CHAT_CACHE[key] = (now, result)
        return result

    def _chat_ollama(self, system: str, user: str, temperature: float, timeout: Optional[float] = None) -> str:
        url = f"{self._ollama_host.rstrip('/')}/api/chat"
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {"temperature": temperature},
        }
        try:
            resp = requests.post(url, json=payload, timeout=timeout or 120)
            if resp.status_code == 404:
                return self._chat_ollama_generate(system=system, user=user, temperature=temperature, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, dict):
                if "message" in data and isinstance(data["message"], dict):
                    content = data["message"].get("content", "")
                    if content:
                        return str(content).strip()
                msgs = data.get("messages")
                if isinstance(msgs, list):
                    texts: List[str] = []
                    for m in msgs:
                        if isinstance(m, dict) and m.get("role") == "assistant" and m.get("content"):
                            texts.append(str(m["content"]))
                    if texts:
                        return "\n".join(texts).strip()
            return str(data).strip()
        except requests.HTTPError as http_err:
            if getattr(http_err.response, "status_code", None) == 404:
                return self._chat_ollama_generate(system=system, user=user, temperature=temperature, timeout=timeout)
            raise

    def _chat_ollama_generate(self, system: str, user: str, temperature: float, timeout: Optional[float] = None) -> str:
        url = f"{self._ollama_host.rstrip('/')}/api/generate"
        prompt = (
            f"[SYSTEM]\n{system}\n\n[USER]\n{user}\n\n[ASSISTANT]"
        )
        payload = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        }
        resp = requests.post(url, json=payload, timeout=timeout or 120)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict) and data.get("response"):
            return str(data["response"]).strip()
        return str(data).strip()


def embed(text: str) -> List[float]:
    now = time.time()
    cached = _EMBED_CACHE.get(text)
    if cached and now - cached[0] < _CACHE_TTL_SECONDS:
        return cached[1]

    host = SETTINGS.ollama_host.rstrip('/')
    model = SETTINGS.ollama_model_embed or SETTINGS.ollama_model
    url_candidates = [
        f"{host}/api/embeddings",
        f"{host}/embeddings",
    ]
    payload = {"model": model, "prompt": text}
    last_err: Optional[Exception] = None
    for url in url_candidates:
        try:
            resp = requests.post(url, json=payload, timeout=60)
            if resp.status_code == 404:
                continue
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, dict):
                vec = data.get("embedding") or data.get("vector")
                if isinstance(vec, list) and vec and isinstance(vec[0], (int, float)):
                    result = [float(x) for x in vec]
                    _EMBED_CACHE[text] = (now, result)
                    return result
            if isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
                first = data["data"][0]
                if isinstance(first, dict) and isinstance(first.get("embedding"), list):
                    result = [float(x) for x in first["embedding"]]
                    _EMBED_CACHE[text] = (now, result)
                    return result
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"Embeddings endpoint not available: {last_err}") 