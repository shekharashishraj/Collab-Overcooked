"""
Provider routing and API-safe kwargs for OpenAI, Anthropic, and OpenAI-compatible servers.
"""
from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Literal, Optional, Tuple

Provider = Literal["openai", "anthropic", "openai_compatible", "human"]


def _anthropic_key_file_candidates(cwd: Optional[str] = None) -> List[str]:
    """Paths to try for anthropic_key.txt (aligns with openai_key discovery in modules)."""
    here = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.dirname(here)
    repo_root = os.path.dirname(src_dir)
    bases = [cwd or os.getcwd(), src_dir, repo_root]
    out: List[str] = []
    seen: set[str] = set()
    for base in bases:
        p = os.path.join(base, "anthropic_key.txt")
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def get_anthropic_api_key(cwd: Optional[str] = None) -> str:
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if key:
        return key
    tried: List[str] = []
    for path in _anthropic_key_file_candidates(cwd):
        tried.append(path)
        if os.path.isfile(path):
            with open(path, "r") as f:
                line = f.readline().strip()
                if line:
                    return line
    raise FileNotFoundError(
        "Anthropic API key not found: set ANTHROPIC_API_KEY or create anthropic_key.txt "
        f"(tried: {', '.join(tried)})"
    )


def infer_provider(model_id: str) -> Provider:
    if not model_id:
        return "openai_compatible"
    m = model_id.strip()
    low = m.lower()
    if "human" in low:
        return "human"
    if low.startswith("claude") or m.startswith("anthropic/"):
        return "anthropic"
    # Official OpenAI-style IDs (incl. DeepSeek via OpenAI-compatible official endpoint)
    if low.startswith("gpt-") or re.match(r"^o\d", low) or low.startswith("o1"):
        return "openai"
    if low.startswith("chatgpt-"):
        return "openai"
    if "deepseek" in low:
        return "openai"
    return "openai_compatible"


def anthropic_api_model_id(model_id: str) -> str:
    """Strip optional anthropic/ prefix for Messages API."""
    m = model_id.strip()
    if m.startswith("anthropic/"):
        return m[len("anthropic/") :]
    return m


def is_reasoning_openai_model(model_id: str) -> bool:
    """
    Models that reject non-default sampling params (temperature, top_p, penalties, stop).
    Heuristic: o-series, gpt-5* except gpt-5-chat*.
    """
    low = model_id.strip().lower()
    if re.match(r"^o\d", low) or low.startswith("o1") or low.startswith("o3") or low.startswith("o4"):
        return True
    if low.startswith("gpt-5") and "chat" not in low:
        return True
    return False


def build_openai_chat_kwargs(
    model_id: str, messages: List[dict], temperature: float
) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {"model": model_id, "messages": messages}
    if is_reasoning_openai_model(model_id):
        return kwargs
    kwargs["temperature"] = temperature
    return kwargs


def _messages_to_anthropic(
    messages: List[dict],
) -> Tuple[Optional[str], List[dict]]:
    """Split leading system message(s) for Anthropic system= ; rest as user/assistant."""
    system_parts: List[str] = []
    rest: List[dict] = []
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "")
        if role == "system":
            system_parts.append(content if isinstance(content, str) else str(content))
        else:
            rest.append({"role": role, "content": content})
    system = "\n\n".join(system_parts) if system_parts else None
    return system, rest


def anthropic_messages_create(
    client: Any,
    model_id: str,
    messages: List[dict],
    temperature: float,
    max_tokens: int,
) -> Any:
    api_model = anthropic_api_model_id(model_id)
    system, anth_msgs = _messages_to_anthropic(messages)
    kwargs: Dict[str, Any] = {
        "model": api_model,
        "max_tokens": max_tokens,
        "messages": anth_msgs,
    }
    if system:
        kwargs["system"] = system
    kwargs["temperature"] = temperature
    try:
        return client.messages.create(**kwargs)
    except TypeError:
        kwargs.pop("temperature", None)
        return client.messages.create(**kwargs)


def extract_human_port_response(response: dict) -> str:
    """Format human UI response dict into the planner text shape."""
    response_template = (
        "{role} analysis: [NOTHING]\n{role} plan: {plan}\n{role} say: {say}"
    )
    if response.get("agent") == "agent1":
        role = "Assistant"
    elif response.get("agent") == "agent0":
        role = "Chef"
    else:
        raise ValueError("Return invalide agent info!")
    response_template = response_template.replace("{role}", role)
    response_template = response_template.replace("{plan}", response.get("plan", ""))
    response_template = response_template.replace(
        "{say}", response["say"] if response.get("say") != "" else "[NOTHING]"
    )
    return response_template


def extract_text(response: Any, provider: Provider) -> str:
    if provider == "human":
        raise ValueError("extract_text not used for human")
    if provider == "anthropic":
        if hasattr(response, "content") and response.content:
            block = response.content[0]
            if hasattr(block, "text"):
                return block.text
        return str(response)
    if isinstance(response, dict):
        if "choices" in response:
            ch0 = response["choices"][0]
            if "message" in ch0:
                return ch0["message"].get("content") or ""
            if "text" in ch0:
                return ch0.get("text") or ""
            return ch0.get("content") or ""
        if "content" in response and response["content"]:
            return response["content"][0].get("text", "")
        return ""
    # OpenAI SDK object
    if hasattr(response, "choices") and response.choices:
        msg = response.choices[0].message
        if msg and getattr(msg, "content", None) is not None:
            return msg.content
    return ""


def openai_output_token_estimate(response: Any) -> Optional[int]:
    if response is None:
        return None
    u = getattr(response, "usage", None)
    if u is None:
        return None
    return getattr(u, "completion_tokens", None) or getattr(
        u, "output_tokens", None
    )


def anthropic_output_token_count(response: Any) -> int:
    u = getattr(response, "usage", None)
    if u is None:
        return 0
    return int(getattr(u, "output_tokens", 0) or 0)
