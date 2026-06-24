"""Tests for Ollama request payload construction and think mode semantics."""

from __future__ import annotations

from llm_auditions.ollama_client import OllamaClient, normalize_think_mode


def test_normalize_think_mode_boolean_values():
    assert normalize_think_mode("false") == (False, "false")
    assert normalize_think_mode("true") == (True, "true")


def test_normalize_think_mode_string_levels():
    assert normalize_think_mode("low") == ("low", "low")
    assert normalize_think_mode("medium") == ("medium", "medium")
    assert normalize_think_mode("high") == ("high", "high")


def test_build_payload_places_think_top_level():
    client = OllamaClient(base_url="http://localhost:11434")
    payload, _ = client.build_chat_payload(
        model="gemma4:12b",
        messages=[{"role": "user", "content": "hello"}],
        think_mode="low",
        structured_output_mode="none",
    )
    assert payload["think"] == "low"
    assert "think" not in payload["options"]
    assert "thinking_budget" not in payload["options"]


def test_build_payload_ollama_json_format():
    client = OllamaClient(base_url="http://localhost:11434")
    payload, _ = client.build_chat_payload(
        model="gemma4:12b",
        messages=[{"role": "user", "content": "hello"}],
        think_mode="false",
        structured_output_mode="ollama_json",
    )
    assert payload["format"] == "json"


def test_build_payload_ollama_schema_format():
    client = OllamaClient(base_url="http://localhost:11434")
    schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
    payload, _ = client.build_chat_payload(
        model="gemma4:12b",
        messages=[{"role": "user", "content": "hello"}],
        think_mode="false",
        structured_output_mode="ollama_schema",
        schema=schema,
    )
    assert payload["format"] == schema
