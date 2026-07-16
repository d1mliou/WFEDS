# WFEDS Phase 1: containerized deterministic pipeline (engine + Python chain).
#
# Build FROM THE REPO ROOT (the context must include scripts/ and docker/):
#     docker build -f docker/Dockerfile -t wfeds:phase1 .
#
# Two stages, BOTH pinned to the SAME base tag. This is deliberate: the engine
# binary and its venv's compiled wheels link against this glibc; running them
# on a different (older) base would fail with "GLIBC_x.y not found". Do not
# "optimize" one stage to a different base image later.
#
# Two SEPARATE venvs, also deliberate:
#   /opt/venv-engine  Cell2Fire's own deps (unpinned numpy -> 2.x)
#   /opt/venv-app     the WFEDS pipeline (numpy<2, pinned by requirements.txt
#                     because the rasterio 1.3.11 ecosystem breaks on numpy 2)
# Installing both into one env is a guaranteed numpy conflict.

# --------------------------------------------------------------------------
# Stage 1: build the patched Cell2Fire engine
# --------------------------------------------------------------------------
FROM ubuntu:24.04 AS engine-builder

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git ca-certificates \
        libboost-all-dev libeigen3-dev \
        python3 python3-venv python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Clean upstream clone at the exact base commit the WFEDS patches target.
RUN git clone https://github.com/cell2fire/Cell2Fire /opt/Cell2Fire \
    && cd /opt/Cell2Fire \
    && git checkout b860bcc

# The two WFEDS engine patches, referenced straight from the repo (single
# source of truth - see scripts/cell2fire/engine_patch/README.md). Order matters.
COPY scripts/cell2fire/engine_patch/cell2fire_initialburned.patch /tmp/patches/
COPY scripts/cell2fire/engine_patch/cell2fire_intensity.patch /tmp/patches/
RUN cd /opt/Cell2Fire \
    && git apply /tmp/patches/cell2fire_initialburned.patch \
    && git apply /tmp/patches/cell2fire_intensity.patch

# FULL rebuild - never incremental. The upstream Makefile's dependency
# tracking is incomplete: after these patches change struct layouts, a plain
# `make` leaves an ABI mismatch (stale .o files read the struct at old
# offsets -> garbage wind/ROS -> the fire silently fails to spread).
# `make clean` is itself broken upstream (nonexistent Forest.o target),
# hence the explicit rm. See scripts/cell2fire/engine_patch/README.md.
RUN cd /opt/Cell2Fire/cell2fire/Cell2FireC \
    && rm -f *.o *.gch \
    && make

# Cell2Fire's own Python wrapper env (its requirements are unpinned - numpy 2.x).
RUN python3 -m venv /opt/venv-engine \
    && /opt/venv-engine/bin/pip install --no-cache-dir -r /opt/Cell2Fire/requirements.txt \
    && /opt/venv-engine/bin/pip install --no-cache-dir -e /opt/Cell2Fire

# --------------------------------------------------------------------------
# Stage 2: runtime - the WFEDS pipeline + the built engine
# --------------------------------------------------------------------------
FROM ubuntu:24.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive
# libgl1/libglib2.0-0: opencv-python (pulled by Cell2Fire's own requirements)
# imports libGL at runtime even headless.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv python3-pip ca-certificates \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# The engine tree + its venv, as built. The whole repo tree is needed (not
# just the binary): Cell2FireC_class.py resolves the binary via a fixed
# relative path inside the repo, and `python -m cell2fire.main` is the same
# entrypoint the WSL path uses today.
COPY --from=engine-builder /opt/Cell2Fire /opt/Cell2Fire
COPY --from=engine-builder /opt/venv-engine /opt/venv-engine

# The WFEDS pipeline env. requirements.txt is the repo's own; matplotlib /
# Pillow / imageio(+ffmpeg) are needed by snapshot_map.py (PNG + MP4 via
# imageio's libx264) but are not listed there - a known, accepted gap in
# that file (see requirements-dev.txt's own comment about litellm).
COPY requirements.txt /tmp/requirements.txt
RUN python3 -m venv /opt/venv-app \
    && /opt/venv-app/bin/pip install --no-cache-dir -r /tmp/requirements.txt \
    && /opt/venv-app/bin/pip install --no-cache-dir \
        matplotlib Pillow imageio imageio-ffmpeg \
        "shapely==2.0.6" \
        litellm orjson fastapi "uvicorn[standard]" \
    && /opt/venv-app/bin/python -c "import rasterio, pyogrio, geopandas, osmnx, imageio, litellm, orjson, fastapi, uvicorn"
# orjson: litellm imports it LAZILY inside completion() (not at import time,
# so the smoke test above alone can't catch its absence) - found live in the
# Phase-2 E2E test as "ModuleNotFoundError: No module named 'orjson'".
# shapely pinned to match the exact version the current WSL host resolves
# (GEOS 3.11.4) - requirements.txt leaves shapely unpinned (an osmnx
# transitive dep), and an unpinned shapely->GEOS drift (2.1.2/GEOS 3.13.1 was
# observed pip-resolved here otherwise) produces a tiny but real polygon-area
# divergence downstream (~0.02% fire_km2, found during Phase-1 verification -
# every raw engine grid was still byte-identical; only the Python-side
# polygonization differed). Pinning here, not in requirements.txt, keeps the
# fix scoped to this new file per the zero-existing-file-changes rule.
# ^ build-time smoke test: the manylinux wheels bundle GDAL/GEOS/PROJ. If this
# line ever fails, add: apt-get install gdal-bin libgdal-dev libgeos-dev libproj-dev

# The pipeline + agent code, read-only copies. NB: scripts/data_prep/ is
# deliberately NOT copied - it holds a second, identical _paths.py whose
# import-name collision with cell2fire's is a documented landmine
# (conftest.py); with only one _paths in the image the issue cannot arise.
COPY scripts/cell2fire/ /app/scripts/cell2fire/
COPY scripts/agent/ /app/scripts/agent/
COPY docker/engine_local_run.py docker/run_container.py docker/rerun_downstream.py docker/api.py /app/docker/
COPY docker/web/ /app/docker/web/

# Data comes from a host mount at run time; LLM keys via --env-file (never
# baked into the image - .env is not COPY'd, and llm_config's own .env loader
# no-ops in here since /app/.env doesn't exist).
#
#   Web server (default CMD):
#     docker run --rm -p 8000:8000 --env-file .env \
#         -v "<host DATA_DIR>:/data" -e WFEDS_DATA_DIR=/data wfeds:phase2
#   Phase-1 CLI (override CMD with the script path):
#     docker run --rm -v "<host DATA_DIR>:/data" -e WFEDS_DATA_DIR=/data \
#         wfeds:phase2 /app/docker/run_container.py --points "38.90,23.12" ...
ENV PYTHONIOENCODING=utf-8

ENTRYPOINT ["/opt/venv-app/bin/python"]
CMD ["/app/docker/api.py"]
