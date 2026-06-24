from __future__ import annotations

from llm_auditions.ollama_client import OllamaClient


def test_transport_payload_modes():
    c = OllamaClient()
    p_prompt, _ = c.build_chat_payload("gemma4:12b", [{"role": "user", "content": "hi"}], structured_output_mode="prompt_only")
    p_json, _ = c.build_chat_payload("gemma4:12b", [{"role": "user", "content": "hi"}], structured_output_mode="ollama_json")
    p_schema, _ = c.build_chat_payload("gemma4:12b", [{"role": "user", "content": "hi"}], structured_output_mode="ollama_schema", schema={"type": "object"})

    assert "format" not in p_prompt
    assert p_json["format"] == "json"
    assert isinstance(p_schema["format"], dict)
