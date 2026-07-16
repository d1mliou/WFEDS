"""LLM provider presets - the agent is MODEL-AGNOSTIC by design.

The agent code talks to one interface (LiteLLM model strings); WHICH model runs
is configuration, never code. Three presets (Gemini / Claude / ChatGPT), default
**gemini-pro**. Selection order:

    1. explicit argument            get_model("claude")
    2. env var                      WFEDS_LLM_PRESET=gpt
    3. DEFAULT_PRESET below         ("gemini-pro")

API keys are NEVER hardcoded or committed: they come from the environment or a
local `.env` file at the repo root (gitignored; see `.env.example`). This also
enables the thesis evaluation axis "same scenario, different models".

Usage:
    from llm_config import get_model
    model, key = get_model()          # -> ("gemini/gemini-2.5-pro", "<key>")
"""

import os
from pathlib import Path

# LiteLLM model strings - plain config, update as providers release new versions.
PRESETS = {
    "gemini-pro": {"model": "gemini/gemini-2.5-pro", "key_env": "GEMINI_API_KEY"},
    "claude":     {"model": "anthropic/claude-sonnet-5", "key_env": "ANTHROPIC_API_KEY"},
    "gpt":        {"model": "openai/gpt-5.1", "key_env": "OPENAI_API_KEY"},
}
DEFAULT_PRESET = "gemini-pro"

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_dotenv():
    """Minimal .env loader (KEY=value lines; no new dependency). Existing env wins."""
    env = _REPO_ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())


def get_model(preset=None):
    """Resolve (litellm_model_string, api_key) for the chosen preset.

    Raises a clear error naming the missing env var - never invents a key."""
    _load_dotenv()
    name = preset or os.environ.get("WFEDS_LLM_PRESET", DEFAULT_PRESET)
    if name not in PRESETS:
        raise ValueError(f"Unknown LLM preset '{name}'. Available: {sorted(PRESETS)}")
    p = PRESETS[name]
    key = os.environ.get(p["key_env"])
    if not key:
        raise RuntimeError(
            f"Missing API key: set {p['key_env']} in the environment or in "
            f"{_REPO_ROOT / '.env'} (see .env.example).")
    return p["model"], key


if __name__ == "__main__":
    for name in PRESETS:
        try:
            model, key = get_model(name)
            print(f"{name:>11}: {model}  (key: ...{key[-4:]})")
        except RuntimeError as e:
            print(f"{name:>11}: {PRESETS[name]['model']}  (NO KEY: {e})")
