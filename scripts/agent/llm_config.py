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
    # `params` are extra kwargs passed straight to litellm.completion for this
    # preset. gpt-5.6-* refuse function tools together with reasoning on
    # /v1/chat/completions ("set reasoning_effort to 'none'"), so tool use here
    # costs the reasoning pass - a real difference from the other presets that
    # must be declared wherever this preset's numbers are reported.
    "gpt":        {"model": "openai/gpt-5.6-luna", "key_env": "OPENAI_API_KEY",
                   "params": {"reasoning_effort": "none"}},
    # A preset is never repointed once a scored run has been published under it,
    # so a second gpt-5.6 model gets its own name rather than replacing the one
    # above (see [[Decision log]] 2026-08-31 on preset-name provenance).
    "gpt-terra":  {"model": "openai/gpt-5.6-terra", "key_env": "OPENAI_API_KEY",
                   "params": {"reasoning_effort": "none"}},
}
DEFAULT_PRESET = "gemini-pro"

# Sampling applies to EVERY preset. It was never set before 2026-09-01, so all
# runs up to then used the provider default (1.0), a creative-writing setting
# for a tool whose whole point is that two identical questions get the same
# answer. Declared here rather than in code so it lands in each run's
# agent_eval_metrics.json (model_params) and can be compared across runs.
SAMPLING = {"temperature": 0.0}

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


def get_params(preset=None):
    """litellm kwargs for the preset: shared sampling plus provider quirks."""
    name = preset or os.environ.get("WFEDS_LLM_PRESET", DEFAULT_PRESET)
    if name not in PRESETS:
        raise ValueError(f"Unknown LLM preset '{name}'. Available: {sorted(PRESETS)}")
    params = dict(SAMPLING)
    params.update(PRESETS[name].get("params", {}))
    return params


if __name__ == "__main__":
    for name in PRESETS:
        extra = get_params(name)
        note = f"  {extra}" if extra else ""
        try:
            model, key = get_model(name)
            print(f"{name:>11}: {model}  (key: ...{key[-4:]}){note}")
        except RuntimeError as e:
            print(f"{name:>11}: {PRESETS[name]['model']}  (NO KEY: {e}){note}")
