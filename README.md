# LLM-WFEDS

**Developing and Evaluating a Geospatial LLM-Agent Prototype for Wildfire Evacuation Decision Support**


---

## 1. What the application is

WFEDS is a decision-support tool for wildfire evacuation, built for an
emergency-management analyst (not the evacuee). Study area: North Evia, Greece
(Istiaia-Aidipsos + Mantoudi-Limni-Agia Anna municipalities).

**What the user does:** opens the web app, marks where fire has been observed on
the map (dropped pins and/or a drawn polygon of the observed front), and asks a
question in plain Greek - e.g. *"τι θα κάψει σε 6 ώρες;"* or *"τρέξε 12 ώρες με
παράθυρο 20 km για αύριο το μεσημέρι"*.

**What the system does:** a fully deterministic geospatial pipeline simulates the
fire hour by hour (Cell2Fire, free-burning = the worst credible case), removes the
road segments the fire cuts, penalises roads near the flames, and re-solves the
evacuation routing of every threatened settlement to its nearest safe refuge at
EVERY hourly time step. An LLM sits only at the workflow-control layer: it extracts
the parameters from the free text, calls the one deterministic tool, and narrates
the result in plain Greek with recommended response measures - always closing with
an advisory/limits statement (the decision belongs to the operational commander,
not the model).

**What the user gets back:**

- two deterministic cards (scenario parameters, numeric results),
- a plain-Greek narration + advisory measures,
- the result played back ON the map (hour slider: fire growth, evacuation routes,
  settlement status routed / cut off / impacted),
- a static map (PNG), an hour-by-hour animation (MP4), an interactive dashboard (HTML),
- 4 GIS files per run (GeoJSON, EPSG:2100): hourly fire zones (polygons), hourly
  fronts (lines), all settlements with their evacuation hour, and the full road
  network with the evacuation segments flagged,
- an automatic per-simulation deliverable folder (`Data/Exports/<run_id>/`) with all
  of the above.

Two interfaces share the same agent: the **web app** (primary, fully containerized)
and a **Telegram bot** (secondary/mobile, runs on the host with the engine in WSL).

## 2. Installation & running, step by step

You need a Windows 10/11 PC (8 GB+ RAM recommended) and an internet connection.
Everything else is installed below.

### Step 1 - Install Docker Desktop

The whole system (fire engine + pipeline + web server) runs inside one Docker
container, so Docker is the only real dependency.

1. Download Docker Desktop for Windows from <https://www.docker.com/products/docker-desktop/>.
2. Run the installer. When asked, keep the default **"Use WSL 2"** option
   (Windows may install/enable WSL automatically; approve it).
3. Restart the PC if the installer asks for it.
4. Start **Docker Desktop** from the Start menu and wait until its whale icon in
   the system tray reports "running".
5. Verify: open **PowerShell** and run
   ```powershell
   docker --version
   ```
   You should see a version number, not an error.

### Step 2 - Get the code

Either clone with git:

```powershell
git clone https://github.com/d1mliou/WFEDS.git
cd WFEDS
```

or, without git: on the GitHub page press **Code → Download ZIP**, extract it,
and open PowerShell inside the extracted folder.

### Step 3 - Get the input data folder

The geodata (~230 MB: DEM, slope, fuel raster, road network, settlements, the 2021
ground truth) is **not** in the repository. Request it from the author and put it
anywhere on your disk, e.g. `C:\WFEDS_Data`. Remember that path - you'll use it in
Step 6. (On the author's own machines the folder lives in OneDrive and is found
automatically; everyone else points to it explicitly.)

### Step 4 - Get an LLM API key and create the .env file

The narration/advisory layer calls an LLM. The default preset uses Google Gemini:

1. Get a free API key at <https://aistudio.google.com/apikey> (Google account needed).
2. In the repo folder, create a plain-text file named exactly `.env` (no .txt
   extension) with one line:
   ```
   GEMINI_API_KEY=το_κλειδί_σου_εδώ
   ```

(Alternatives: `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` plus a second line
`WFEDS_LLM_PRESET=claude` or `gpt`.)

### Step 5 - Build the container image (once)

In PowerShell, inside the repo folder:

```powershell
docker build -f docker/Dockerfile -t wfeds:phase2 .
```

First build takes ~10 minutes: it downloads Ubuntu, clones + patches + compiles
the Cell2Fire engine, and installs the Python stack - all inside the image.
Re-running it later is nearly instant unless the code changed.

### Step 6 - Start the server

**If your data folder is the author's OneDrive setup:** just double-click
`docker\start_server.bat` - it starts Docker Desktop if needed, finds the data
folder, launches the server and opens the browser.

**Otherwise (data at a custom path, e.g. `C:\WFEDS_Data`):** run in PowerShell

```powershell
docker run -d --name wfeds_web -p 8000:8000 --env-file .env `
  -v "C:\WFEDS_Data:/data" -e WFEDS_DATA_DIR=/data wfeds:phase2
```

Then open **http://localhost:8000** in any browser.

### Step 7 - Use it

Drop a pin (or draw a polygon) on the map where fire was observed, type a request
in Greek (e.g. «τι θα κάψει σε 6 ώρες;») and send. A simulation takes ~1-3 minutes;
the reply arrives in the chat panel and the result plays back on the map. Weather
is fetched live per scenario, so the container needs internet while running.

### Stopping / restarting

```powershell
docker stop wfeds_web      # stop
docker start wfeds_web     # start again later (or re-run Step 6)
```

### If something goes wrong

| Symptom | Fix |
|---|---|
| `docker: error during connect ...` | Docker Desktop isn't running - start it and wait for "running" |
| Port 8000 already in use | `docker rm -f wfeds_web` and run Step 6 again, or map another port (`-p 8080:8000` → http://localhost:8080) |
| Chat returns a 500 about the LLM key | the `.env` is missing/misnamed or the key variable doesn't match the preset (Step 4) |
| "data directory does not exist" on startup | the `-v` path in Step 6 doesn't point at the data folder |

### Optional: the Telegram channel

`scripts/agent/start_bot.bat` (needs `TELEGRAM_BOT_TOKEN` in `.env`, Python on the
host, and a WSL Cell2Fire build - the pre-Docker setup; see branch `legacy-main`
for that system in its original form).

### Optional: run the test suite (no Docker, data or keys needed)

```powershell
pip install -r requirements.txt -r requirements-dev.txt
python -m pytest          # 237 tests, all mocked - no real engine/network
```

## Under the hood (brief)

| Component | Description |
|---|---|
| Fire spread | **Cell2Fire** - the actual fire-spread simulator, run hour by hour, free-burning (worst case: no firefighting assumed). Its source code is not copied into this repository. Instead, the repo only records which exact version of the original Cell2Fire project to use (commit `b860bcc`) plus two small changes made for this thesis, in `scripts/cell2fire/engine_patch/` (let the fire start from an observed front line; report flame intensity). Every time the Docker image is built, that exact version is automatically downloaded and compiled from scratch - checked to produce identical results to the original (non-Docker) setup it replaced. |
| Network & exposure | Edges crossing the perimeter removed; survivors friction-penalised by distance; per-hour re-routing to nearest safe refuge; settlements flagged routed / cut off / impacted. |
| LLM layer | One coarse tool wraps the whole pipeline; user times are Greece-local (the tool owns the UTC conversion); geometry only ever comes from the user (pins/polygon - never guessed). |
| Web backend | FastAPI in the same container: cookie sessions, SSE progress, GIS export generation, same-origin UI (vanilla JS + Leaflet). |

```
scripts/cell2fire/   the deterministic pipeline (fire sim, evacuation, dashboard)
scripts/agent/       the LLM agent + Telegram channel
scripts/data_prep/   one-off geodata acquisition
docker/              container build, FastAPI backend, browser UI, verification tooling
tests/               pytest suite
```

Branches: `main` = current system. `legacy-main` = frozen pre-Docker (WSL/Telegram-only) state.

## Next steps

The implementation is complete end to end; the remaining thesis work is evaluation
and write-up: **system-level validation** of the agent (parameter extraction,
tool-call sequencing, faithfulness of the narration to the tool outputs - the design
of this evaluation is not yet fixed), scenario-comparison metrics, and a bounded
sensitivity analysis on the project's own parameters.
