"""`dtreat serve` — FastAPI debug server + static UI.

One view per pipeline stage: browse artifacts, see the distributions, drill
into single responses/judgments, and replay permutation nulls. Everything is
read-only over the run directory.
"""

from __future__ import annotations

from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, PlainTextResponse

from dtreat.common.console_logging import log

from . import run_data_api

STATIC_DIR = Path(__file__).parent / "static"


def create_debug_app(runs_root: Path) -> FastAPI:
    app = FastAPI(title="dtreat debug server", docs_url="/api/docs")
    runs_root = Path(runs_root)

    def _paths(run_name: str):
        try:
            return run_data_api.paths_for(runs_root, run_name)
        except ValueError as error:
            raise HTTPException(400, str(error)) from error

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "debug_ui.html")

    @app.get("/static/{filename}")
    def static_file(filename: str) -> FileResponse:
        target = (STATIC_DIR / filename).resolve()
        if not str(target).startswith(str(STATIC_DIR.resolve())) or not target.exists():
            raise HTTPException(404, "not found")
        return FileResponse(target)

    @app.get("/api/runs")
    def runs() -> list[dict]:
        return run_data_api.list_runs(runs_root)

    @app.get("/api/runs/{run_name}/overview")
    def overview(run_name: str) -> dict:
        return run_data_api.run_overview(_paths(run_name))

    @app.get("/api/runs/{run_name}/stage1")
    def stage1(run_name: str) -> dict:
        return _load(run_data_api.stage1_data, _paths(run_name))

    @app.get("/api/runs/{run_name}/stage2")
    def stage2(run_name: str) -> dict:
        return _load(run_data_api.stage2_data, _paths(run_name))

    @app.get("/api/runs/{run_name}/stage3")
    def stage3(
        run_name: str,
        community: str | None = None,
        search: str | None = None,
        limit: int = Query(50, le=500),
        offset: int = 0,
    ) -> dict:
        return _load(
            run_data_api.stage3_data, _paths(run_name), community, search, limit, offset
        )

    @app.get("/api/runs/{run_name}/stage4")
    def stage4(
        run_name: str,
        limit: int = Query(50, le=500),
        offset: int = 0,
        community: str | None = None,
        axis_id: str | None = None,
        verdict: str | None = None,
        search: str | None = None,
    ) -> dict:
        return _load(
            run_data_api.stage4_data, _paths(run_name), limit, offset,
            community, axis_id, verdict, search,
        )

    @app.get("/api/runs/{run_name}/response/{response_id}")
    def response_detail(run_name: str, response_id: str) -> dict:
        return _load(run_data_api.response_detail, _paths(run_name), response_id)

    @app.get("/api/runs/{run_name}/stage5")
    def stage5(run_name: str) -> dict:
        return _load(run_data_api.stage5_data, _paths(run_name))

    @app.get("/api/runs/{run_name}/permutation-null/{axis_id}")
    def permutation_null(run_name: str, axis_id: str) -> dict:
        return _load(run_data_api.permutation_null_for_axis, _paths(run_name), axis_id)

    @app.get("/api/runs/{run_name}/trace")
    def trace(
        run_name: str,
        grep: str | None = None,
        errors_only: bool = False,
        limit: int = Query(200, le=2000),
    ) -> dict:
        return run_data_api.trace_data(_paths(run_name), grep, errors_only, limit)

    @app.get("/api/runs/{run_name}/summary.md", response_class=PlainTextResponse)
    def summary_markdown(run_name: str) -> str:
        path = _paths(run_name).analysis_summary_path
        if not path.exists():
            raise HTTPException(404, "no analysis summary yet")
        return path.read_text()

    return app


def _load(fn, *args):
    """Translate missing-artifact errors into clean 404s."""
    try:
        return fn(*args)
    except FileNotFoundError as error:
        raise HTTPException(404, f"artifact not written yet: {error}") from error
    except ValueError as error:
        raise HTTPException(400, str(error)) from error


def serve_debug_ui(runs_root: Path, port: int) -> None:
    app = create_debug_app(runs_root)
    log(f"dtreat debug server on http://127.0.0.1:{port}  (runs root: {runs_root})")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
