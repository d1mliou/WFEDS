# WFEDS web/Docker

One image, two layers:

- **The deterministic core (verified):** the Cell2Fire engine + the whole
  deterministic chain run inside one Linux container, producing output
  equivalent to the original Windows+WSL path.
- **The web backend (this image's default mode):** a FastAPI backend exposing
  the UNMODIFIED `WfedsAgent` (the same agent the Telegram bot uses) over
  HTTP - pins, chat with SSE progress, and file serving - for the browser
  frontend.

**Zero pre-existing pipeline files are modified** - the pre-Docker WSL/Telegram
setup (frozen on branch `legacy-main`) keeps working unchanged and remains the
fallback.

How it works: both entrypoints import the *unmodified*
`scripts/cell2fire/run_scenario.py` and swap ONE module global - `run_engine`
(the WSL bridge) - for `engine_local_run.run_engine_docker` (same contract,
native in-container subprocess). Every other stage, and the whole agent
layer, runs the exact same code as on the host. This late-binding patch is
the same mechanism the repo's own tests use
(`tests/cell2fire/test_run_scenario.py`).

## Build (from the REPO ROOT, not from docker/)

```powershell
docker build -f docker/Dockerfile -t wfeds .
```

The Cell2Fire engine source is NOT part of this repository: the first build
downloads the original Cell2Fire project at the exact version this thesis
targets (commit `b860bcc`), applies the two small WFEDS changes from
`scripts/cell2fire/engine_patch/*.patch`, and compiles it from scratch
(full `rm -f *.o *.gch && make` rebuild) - takes a while; cached afterwards.

> **Git Bash gotcha (hit on the first real run):** MSYS rewrites
> Unix-looking args, so `-e WFEDS_DATA_DIR=/data` reaches docker as
> `C:/Program Files/Git/data` and `_paths.py` fails with a "does not exist
> on disk" error. Prefix every `docker run` with `MSYS_NO_PATHCONV=1` when
> running from Git Bash; PowerShell needs nothing.

## Run the web server - one click

Double-click **`docker/start_server.bat`**: starts Docker Desktop if needed,
resolves the data folder via the OneDrive env vars (machine-independent, same
logic as `_paths.py`), (re)starts the `wfeds_web` container and opens the
browser. Stop with `docker stop wfeds_web`.

## Run the web server manually (what the .bat does)

```powershell
# resolve your DATA_DIR first (what _paths.py resolves to), e.g.:
#   python -c "import sys; sys.path.insert(0,'scripts/cell2fire'); from _paths import DATA_DIR; print(DATA_DIR)"

docker run --rm -p 8000:8000 --env-file .env `
  -v "<DATA_DIR>:/data" `
  -e WFEDS_DATA_DIR=/data `
  wfeds
```

- **API keys:** supplied ONLY at run time via `--env-file .env` (Docker parses
  the file on the HOST and injects real env vars - this is unrelated to
  `llm_config.py`'s own `.env` loader, which no-ops in the container since
  `/app/.env` is never COPY'd; real env vars win in its `setdefault` logic
  anyway). Only the active preset's key is needed - `GEMINI_API_KEY` for the
  default `gemini-pro` preset (or set `WFEDS_LLM_PRESET=claude|gpt` plus its
  key). `TELEGRAM_BOT_TOKEN` is NOT needed in web mode.
- Weather is fetched live from Open-Meteo at run time - the container needs
  network egress.

### Endpoints

| Endpoint | Purpose |
|---|---|
| `POST /api/pins` `{lat, lon}` | register a fire-observation pin; returns the pin-time geometry disclosure (one front vs N separate fires - same `_n_fires` rule as the Telegram bot) |
| `POST /api/clear` | forget the session's pins + conversation |
| `GET /api/state` | current pins + whether a chat is running |
| `POST /api/chat` `{text}` | start a chat turn → `202 {stream_id}`; `409` if one is already running for this session; `500` if the LLM key/preset is misconfigured |
| `GET /api/chat/{stream_id}/stream` | SSE: `progress` event(s) → long silence while the engine runs (1–15 min) → one `done` `{reply, cards, files}` or `error` event |
| `GET /files/{run_id}/{filename}` | serve `map.png` / `map.mp4` / `fire_timesteps.html` for a finished run (whitelisted names only) |

Sessions are uuid-cookie-keyed and in-memory (lost on container restart -
same accepted tradeoff as the Telegram bot's per-chat dicts). `/files/` has
no auth - single-user local prototype, stated non-goal.

## Run the pipeline CLI, no web/LLM (override the CMD)

The ENTRYPOINT is now the bare venv python; pass the script path explicitly:

```powershell
docker run --rm `
  -v "<DATA_DIR>:/data" `
  -e WFEDS_DATA_DIR=/data `
  wfeds /app/docker/run_container.py --points "38.90,23.12" --window-km 12 --horizon 6 --start 2021-08-06T10:00
```

Same CLI as `run_scenario.py` (`--scenario north_evia_2021`, `--label`,
`--run-id`, ...). Runs land under `Fire/cell2fire/runs/<run_id>/` in the
mounted data folder, visible on the host. No API keys needed for this path.

## Verify equivalence (container vs WSL ground truth)

1. **Reference run (WSL path, on the host as today)** with a pinned run id:
   ```powershell
   python scripts/cell2fire/run_scenario.py --points "38.90,23.12" --window-km 6 --horizon 2 --start 2021-08-06T10:00 --run-id ref_smoke
   ```
2. **Same inputs through the container** (different run id):
   ```powershell
   docker run --rm -v "<DATA_DIR>:/data" -e WFEDS_DATA_DIR=/data `
     wfeds /app/docker/run_container.py --points "38.90,23.12" --window-km 6 --horizon 2 --start 2021-08-06T10:00 --run-id cand_smoke
   ```
3. **Compare** (grids cell-by-cell + result.json, volatile fields excluded):
   ```powershell
   python docker/verify_equivalence.py --reference "<DATA_DIR>/Fire/cell2fire/runs/ref_smoke" --candidate "<DATA_DIR>/Fire/cell2fire/runs/cand_smoke"
   ```

Success = `EQUIVALENT`. Byte-identical grids are expected/required; summary
floats (`fire_km2`, `longest_route_km`) tolerate 1% relative drift (GDAL/GEOS
patch-version float noise), exact-count fields must match exactly. Tier 2
(full golden): repeat with `--scenario north_evia_2021` (historical ERA5 =
stable inputs; ~15 min per side).

**Status 2026-07-15 - BOTH tiers PASSED.**
- Tier 1 smoke (1 pin, 6 km / 2 h): every grid + every summary field
  byte-identical, zero float drift.
- Tier 2 golden (`north_evia_2021`, full 24 h): all 50 grids byte-identical.
  First pass caught a REAL divergence downstream (unpinned `shapely` resolved
  2.0.6/GEOS 3.11.4 on WSL vs 2.1.2/GEOS 3.13.1 in-container → ~0.02–0.16%
  polygon-area drift); fixed by pinning `shapely==2.0.6` in the image. Final
  verdict: `EQUIVALENT`.

## Known constraints

- Both Dockerfile stages are pinned to the SAME base tag (`ubuntu:24.04`) -
  glibc compatibility between the engine build and runtime. Don't diverge them.
- Two venvs inside the image (`/opt/venv-engine` numpy 2.x vs `/opt/venv-app`
  numpy<2) - mandatory, not stylistic; see the Dockerfile header.
- `shapely` is pinned (2.0.6, in `pyproject.toml`) to match the verified
  setup - an unpinned resolve drifts GEOS and produces real polygon-area
  differences.
- `Messages/` engine output is intentionally NOT copied back (the WSL path
  never did either; the adapter skips the ROS grid) - equivalence means
  matching today's behaviour exactly.
- `scripts/data_prep/` is deliberately never COPY'd into the image (avoids
  the documented `_paths.py` import-name collision).
