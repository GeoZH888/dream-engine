"""Shared Claude call: schema-validated JSON output, refusal handling, optional fallback."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


class LLMError(RuntimeError):
    pass


def has_credentials() -> bool:
    """True if the Anthropic SDK can likely authenticate (env key/token or an `ant` profile)."""
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return True
    return (Path.home() / ".config" / "anthropic").exists()


def make_client() -> Any:
    import anthropic  # the key is read from the environment, never hard-coded

    return anthropic.Anthropic(max_retries=6)


def structured_call(client: Any, cfg: dict[str, Any], *, system: str, user: str, schema: dict,
                    effort: str, temperature: float | None = None) -> dict:
    """One Messages API call constrained to ``schema``; returns the parsed JSON object."""
    llm = cfg["llm"]
    kwargs: dict[str, Any] = dict(
        model=cfg["model"],
        max_tokens=llm["max_tokens"],
        system=system,
        messages=[{"role": "user", "content": user}],
        output_config={"effort": effort, "format": {"type": "json_schema", "schema": schema}},
    )
    if temperature is not None and llm.get("send_temperature"):
        kwargs["temperature"] = temperature
    if llm.get("fallbacks"):
        response = client.beta.messages.create(
            betas=["server-side-fallback-2026-07-01"], fallbacks=llm["fallbacks"], **kwargs)
    else:
        response = client.messages.create(**kwargs)

    if response.stop_reason == "refusal":
        raise LLMError("The model declined this request.")
    if response.stop_reason == "max_tokens":
        raise LLMError("Response hit max_tokens; raise llm.max_tokens.")
    text = next(b.text for b in response.content if b.type == "text")
    return json.loads(text)
