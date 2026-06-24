"""Ollama API client."""

from __future__ import annotations

import logging
import time
import subprocess
from typing import Any, Optional

import requests

from .models import ModelInfo, ModelResponse, OllamaMetrics

logger = logging.getLogger(__name__)

def normalize_think_mode(think_mode: str | bool) -> tuple[Any, str]:
    """
    Map configured think mode to exact payload value:
    - "false" -> False
    - "true" -> True
    - "low"|"medium"|"high" -> same string
    """
    if isinstance(think_mode, bool):
        return think_mode, str(think_mode).lower()
    mode = str(think_mode).strip().lower()
    if mode == "false":
        return False, "false"
    if mode == "true":
        return True, "true"
    if mode in {"low", "medium", "high"}:
        return mode, mode
    return False, mode


class OllamaClient:
    """Thin client for the Ollama local API."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        timeout: Optional[float] = None,
        keep_alive: int = 0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.keep_alive = keep_alive
        self._session = requests.Session()

    # ------------------------------------------------------------------
    # Model inventory
    # ------------------------------------------------------------------

    def list_models(self) -> list[ModelInfo]:
        """Fetch installed models from Ollama API."""
        try:
            resp = self._session.get(
                f"{self.base_url}/api/tags",
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.error("Failed to list models from Ollama: %s", exc)
            return []

        models: list[ModelInfo] = []
        for m in data.get("models", []):
            details = m.get("details", {})
            full_digest = m.get("digest", "")
            models.append(
                ModelInfo(
                    name=m.get("name", ""),
                    id=full_digest[:12],
                    full_digest=full_digest,
                    size=str(round(m.get("size", 0) / 1e9, 1)),
                    modified=m.get("modified_at", ""),
                    family=details.get("family", ""),
                    parameter_size=details.get("parameter_size", ""),
                    quantization_level=details.get("quantization_level", ""),
                )
            )
        return models

    def show_model(self, model: str) -> dict[str, Any]:
        """Fetch /api/show metadata for a model."""
        resp = self._session.post(
            f"{self.base_url}/api/show",
            json={"name": model},
            timeout=20,
        )
        resp.raise_for_status()
        return resp.json()

    def discover_model_capabilities(self) -> list[ModelInfo]:
        """Combine /api/tags inventory with /api/show details per model."""
        models = self.list_models()
        enriched: list[ModelInfo] = []
        for m in models:
            supports_thinking = False
            supports_vision = False
            raw_show: dict[str, Any] = {}
            family = m.family
            parameter_size = m.parameter_size
            quantization_level = m.quantization_level
            capabilities: list[str] = []
            try:
                raw_show = self.show_model(m.name)
                modelinfo = raw_show.get("model_info", {}) if isinstance(raw_show, dict) else {}
                template = str(raw_show.get("template", ""))
                cap_obj = raw_show.get("capabilities")
                if isinstance(cap_obj, dict):
                    for k, v in cap_obj.items():
                        if bool(v):
                            capabilities.append(str(k))
                elif isinstance(cap_obj, list):
                    capabilities.extend(str(c) for c in cap_obj)
                # Conservative detection from live metadata only
                cap_blob = (" ".join(capabilities) + " " + str(modelinfo) + " " + template).lower()
                supports_thinking = any(tok in cap_blob for tok in ("think", "reason", "reasoning"))
                supports_vision = any(tok in cap_blob for tok in ("vision", "image", "multimodal"))
                if "general.architecture" in modelinfo:
                    family = str(modelinfo.get("general.architecture", family))
            except Exception as exc:
                logger.warning("/api/show failed for %s: %s", m.name, exc)

            enriched.append(
                ModelInfo(
                    name=m.name,
                    id=m.id,
                    full_digest=m.full_digest or m.id,
                    size=m.size,
                    modified=m.modified,
                    family=family,
                    parameter_size=parameter_size,
                    quantization_level=quantization_level,
                    capabilities=sorted(set(capabilities)),
                    supports_thinking=supports_thinking,
                    supports_vision=supports_vision,
                    raw_show=raw_show,
                )
            )
        return enriched

    def build_chat_payload(
        self,
        model: str,
        messages: list[dict[str, Any]],
        think_mode: str = "false",
        num_ctx: int = 8192,
        num_predict: int = 4096,
        temperature: float = 0.0,
        stream: bool = False,
        structured_output_mode: str = "prompt_only",
        schema: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], str]:
        think_value, mode_norm = normalize_think_mode(think_mode)
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": stream,
            "think": think_value,
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": temperature,
                "num_ctx": num_ctx,
                "num_predict": num_predict,
            },
        }
        if structured_output_mode == "ollama_json":
            payload["format"] = "json"
        elif structured_output_mode == "ollama_schema" and schema is not None:
            payload["format"] = schema
        return payload, mode_norm

    def get_version(self) -> str:
        """Return Ollama version string."""
        try:
            resp = self._session.get(f"{self.base_url}/api/version", timeout=5)
            resp.raise_for_status()
            return resp.json().get("version", "unknown")
        except Exception:
            try:
                result = subprocess.run(
                    ["ollama", "--version"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                return result.stdout.strip() or result.stderr.strip()
            except Exception:
                return "unknown"

    def ollama_ps(self) -> str:
        """Return output of `ollama ps`."""
        try:
            result = subprocess.run(
                ["ollama", "ps"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return (result.stdout or "") + (result.stderr or "")
        except Exception as exc:
            return f"ollama ps failed: {exc}"

    # ------------------------------------------------------------------
    # Chat
    # ------------------------------------------------------------------

    def chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        think_mode: str = "false",
        num_ctx: int = 8192,
        num_predict: int = 4096,
        temperature: float = 0.0,
        stream: bool = False,
        structured_output_mode: str = "prompt_only",
        schema: dict[str, Any] | None = None,
    ) -> ModelResponse:
        """Send a chat request and return a ModelResponse."""
        payload, mode_norm = self.build_chat_payload(
            model=model,
            messages=messages,
            think_mode=think_mode,
            num_ctx=num_ctx,
            num_predict=num_predict,
            temperature=temperature,
            stream=stream,
            structured_output_mode=structured_output_mode,
            schema=schema,
        )

        raw_json: dict[str, Any] = {}
        content = ""
        thinking = ""
        error: Optional[str] = None
        metrics = OllamaMetrics()
        effective_think_mode = mode_norm
        think_mode_accepted = True
        started = time.perf_counter()

        try:
            resp = self._session.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self.timeout,
            )
            if resp.status_code >= 400:
                body_text = resp.text
                if resp.status_code in (400, 422) and "think" in body_text.lower():
                    think_mode_accepted = False
                    raw_json = {"_think_mode_unsupported": True, "error": body_text[:1000]}
                else:
                    resp.raise_for_status()
            if not raw_json:
                resp.raise_for_status()
                raw_json = resp.json()

            # Extract content
            msg = raw_json.get("message", {})
            content = msg.get("content", "")
            thinking = msg.get("thinking", "")

            # Extract metrics
            metrics = OllamaMetrics(
                prompt_eval_count=raw_json.get("prompt_eval_count", 0),
                prompt_eval_duration_ns=raw_json.get("prompt_eval_duration", 0),
                eval_count=raw_json.get("eval_count", 0),
                eval_duration_ns=raw_json.get("eval_duration", 0),
                load_duration_ns=raw_json.get("load_duration", 0),
                total_duration_ns=raw_json.get("total_duration", 0),
                wall_clock_seconds_local=max(0.0, time.perf_counter() - started),
                done=raw_json.get("done", False),
                done_reason=raw_json.get("done_reason", ""),
            )

        except requests.HTTPError as exc:
            error = f"HTTP {exc.response.status_code}: {exc.response.text[:500]}"
            logger.error("Ollama HTTP error for model %s: %s", model, error)
        except requests.Timeout:
            error = "Request timed out"
            logger.error("Ollama timeout for model %s", model)
        except Exception as exc:
            error = str(exc)
            logger.error("Ollama error for model %s: %s", model, exc)

        if metrics.wall_clock_seconds_local <= 0:
            metrics.wall_clock_seconds_local = max(0.0, time.perf_counter() - started)

        return ModelResponse(
            model=model,
            requested_think_mode=str(think_mode),
            effective_think_mode=effective_think_mode,
            think_mode_accepted=think_mode_accepted,
            raw_json=raw_json,
            request_payload=payload,
            content=content,
            thinking=thinking,
            has_thinking_content=bool(str(thinking).strip()),
            truncated_length_stop=metrics.done_reason == "length",
            empty_final_content=not bool(str(content).strip()),
            metrics=metrics,
            error=error,
        )
