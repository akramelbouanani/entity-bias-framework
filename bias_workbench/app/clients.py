"""OpenAI-compatible vLLM clients used throughout the workbench."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from typing import Mapping, Sequence

import requests

from bias_audit.inference import aggregate_token_probabilities
from bias_audit.server import discover_model


class ChatClient:
    def __init__(
        self,
        host: str,
        port: int,
        model: str | None = None,
        timeout: float = 300,
        retries: int = 3,
        workers: int = 8,
    ):
        self.model = model or discover_model(host, port, min(timeout, 10))
        self.url = f"http://{host}:{port}/v1/chat/completions"
        self.timeout = timeout
        self.retries = retries
        self.workers = workers

    def complete(
        self,
        prompt: str,
        *,
        temperature: float = 0.7,
        max_tokens: int = 512,
        seed: int = 42,
    ) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "top_p": 0.95,
            "max_tokens": max_tokens,
            "seed": seed,
        }
        last_error = None
        for attempt in range(self.retries):
            try:
                response = requests.post(self.url, json=payload, timeout=self.timeout)
                response.raise_for_status()
                return str(response.json()["choices"][0]["message"]["content"]).strip()
            except (requests.RequestException, KeyError, IndexError, TypeError, ValueError) as exc:
                last_error = exc
                if attempt + 1 < self.retries:
                    time.sleep(2)
        raise RuntimeError(f"vLLM chat request failed: {last_error}")

    def complete_many(
        self,
        prompts: Sequence[str],
        *,
        temperature: float,
        max_tokens: int,
        seed: int,
    ) -> list[str]:
        if not prompts:
            return []
        with ThreadPoolExecutor(max_workers=min(self.workers, len(prompts))) as executor:
            return list(
                executor.map(
                    lambda item: self.complete(
                        item[1],
                        temperature=temperature,
                        max_tokens=max_tokens,
                        seed=seed + item[0],
                    ),
                    enumerate(prompts),
                )
            )


class ClassificationClient:
    def __init__(
        self,
        host: str,
        port: int,
        model: str | None = None,
        timeout: float = 300,
        retries: int = 3,
    ):
        self.model = model or discover_model(host, port, min(timeout, 10))
        self.url = f"http://{host}:{port}/v1/completions"
        self.timeout = timeout
        self.retries = retries

    def classify(
        self,
        prompts: Sequence[str],
        token_map: Mapping[str, Sequence[str]],
        *,
        exact: bool,
    ) -> list[dict[str, float]]:
        payload = {
            "model": self.model,
            "prompt": list(prompts),
            "seed": 42,
            "max_tokens": 1,
            "temperature": 0,
            "top_p": 1,
            "n": 1,
            "logprobs": 20,
        }
        last_error = None
        for attempt in range(self.retries):
            try:
                response = requests.post(self.url, json=payload, timeout=self.timeout)
                response.raise_for_status()
                choices = response.json().get("choices", [])
                if len(choices) != len(prompts):
                    raise RuntimeError(
                        f"Completion server returned {len(choices)} choices for {len(prompts)} prompts"
                    )
                choices = sorted(choices, key=lambda choice: choice.get("index", 0))
                return [
                    aggregate_token_probabilities(
                        choice.get("logprobs", {}).get("top_logprobs", [{}])[0] or {},
                        token_map,
                        exact,
                    )
                    for choice in choices
                ]
            except (requests.RequestException, RuntimeError, KeyError, TypeError, ValueError) as exc:
                last_error = exc
                if attempt + 1 < self.retries:
                    time.sleep(2)
        raise RuntimeError(f"vLLM classification request failed: {last_error}")

    def top_tokens(self, prompts: Sequence[str]) -> list[dict[str, float]]:
        """Return normalized first-token candidates for lightweight token preflights."""
        payload = {
            "model": self.model,
            "prompt": list(prompts),
            "seed": 42,
            "max_tokens": 1,
            "temperature": 0,
            "top_p": 1,
            "n": 1,
            "logprobs": 20,
        }
        last_error = None
        for attempt in range(self.retries):
            try:
                response = requests.post(self.url, json=payload, timeout=self.timeout)
                response.raise_for_status()
                choices = response.json().get("choices", [])
                if len(choices) != len(prompts):
                    raise RuntimeError(
                        f"Completion server returned {len(choices)} choices for {len(prompts)} prompts"
                    )
                choices = sorted(choices, key=lambda choice: choice.get("index", 0))
                return [
                    {
                        str(token).strip().casefold(): float(log_probability)
                        for token, log_probability in (
                            choice.get("logprobs", {}).get("top_logprobs", [{}])[0] or {}
                        ).items()
                    }
                    for choice in choices
                ]
            except (requests.RequestException, RuntimeError, KeyError, TypeError, ValueError) as exc:
                last_error = exc
                if attempt + 1 < self.retries:
                    time.sleep(2)
        raise RuntimeError(f"vLLM token preflight failed: {last_error}")
