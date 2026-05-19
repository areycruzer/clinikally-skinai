"""FastAPI application factory and configuration for the Clinikally SkinAI backend.

This module assembles the top-level :class:`~fastapi.FastAPI` application,
wiring together:

* **Routers** – modular endpoint groups for queries, collections, user/tree
  config, feedback, tools, utilities, and database operations.
* **Middleware** – permissive CORS (suitable for development / single-origin
  deployments) and centralised error handlers.
* **Background schedulers** – periodic jobs for tree-timeout eviction,
  Weaviate client health checks, and resource usage snapshots.
* **Static file serving** – serves a pre-built Next.js frontend from the
  ``static/`` directory, with a catch-all ``/`` route for SPA deep-links.

The application is created at module level (``app``) so that ASGI servers
(e.g. ``uvicorn skinai.api.app:app``) can import it directly.
"""

import os
from contextlib import asynccontextmanager

from apscheduler.schedulers.asyncio import AsyncIOScheduler

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from skinai.api.core.log import logger, set_log_level
from skinai.api.dependencies.common import get_user_manager
from skinai.api.middleware.error_handlers import register_error_handlers
from skinai.api.routes import (
    collections,
    feedback,
    init,
    processor,
    query,
    user_config,
    tree_config,
    utils,
    tools,
    db,
)
from skinai.api.services.user import UserManager
from skinai.api.utils.resources import print_resources


from pathlib import Path


async def check_timeouts() -> None:
    """Evict idle decision trees that have exceeded their inactivity timeout."""
    user_manager = get_user_manager()
    await user_manager.check_all_trees_timeout()


async def output_resources() -> None:
    """Snapshot current resource utilisation metrics and persist to file."""
    user_manager = get_user_manager()
    await print_resources(user_manager, save_to_file=True)


async def check_restart_clients() -> None:
    """Restart any Weaviate client connections that have become unhealthy."""
    user_manager = get_user_manager()
    await user_manager.check_restart_clients()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager.

    On startup, initialises the :class:`AsyncIOScheduler` with three periodic
    background jobs (using prime-number intervals to minimise scheduling
    collisions):

    * **check_timeouts** – every 29 s – evicts idle decision trees.
    * **check_restart_clients** – every 31 s – heals Weaviate connections.
    * **output_resources** – every ~18 min – logs resource snapshots.

    On shutdown, the scheduler is stopped and all Weaviate clients are closed
    gracefully.
    """
    user_manager = get_user_manager()

    scheduler = AsyncIOScheduler()
    set_log_level("INFO")

    # use prime numbers for intervals so they don't overlap
    scheduler.add_job(check_timeouts, "interval", seconds=29)
    scheduler.add_job(check_restart_clients, "interval", seconds=31)
    scheduler.add_job(output_resources, "interval", seconds=1103)

    scheduler.start()
    yield
    scheduler.shutdown()

    await user_manager.close_all_clients()


# Create FastAPI app instance
app = FastAPI(title="SkinAI API", version="0.3.0", lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register error handlers
register_error_handlers(app)

# Include routers
app.include_router(init.router, prefix="/init", tags=["init"])
app.include_router(query.router, prefix="/ws", tags=["websockets"])
app.include_router(processor.router, prefix="/ws", tags=["websockets"])
app.include_router(collections.router, prefix="/collections", tags=["collections"])
app.include_router(user_config.router, prefix="/user/config", tags=["user config"])
app.include_router(tree_config.router, prefix="/tree/config", tags=["tree config"])
app.include_router(feedback.router, prefix="/feedback", tags=["feedback"])
app.include_router(utils.router, prefix="/util", tags=["utilities"])
app.include_router(tools.router, prefix="/tools", tags=["tools"])
app.include_router(db.router, prefix="/db", tags=["db"])


# Health check endpoint (kept in main app.py due to its simplicity)
@app.get("/api/health", tags=["health"])
async def health_check() -> dict:
    """Liveness probe returning ``{"status": "healthy"}``.

    Used by container orchestrators and uptime monitors to verify that the
    API process is accepting requests.
    """
    logger.info("Health check requested")
    return {"status": "healthy"}


# Mount the app from static files
BASE_DIR = Path(__file__).resolve().parent

# Serve NextJS _next assets at root level (this is crucial!)
app.mount(
    "/_next",
    StaticFiles(directory=BASE_DIR / "static/_next"),
    name="next-assets",
)

# Serve other static files
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="app")


@app.get("/")
@app.head("/")
async def serve_frontend():
    """Catch-all route serving the Next.js SPA entry point.

    Returns ``static/index.html`` if the pre-built frontend exists, enabling
    client-side routing for all non-API paths.  Returns ``None`` (204) when
    no frontend build is present (e.g. API-only development mode).
    """
    if os.path.exists(os.path.join(BASE_DIR, "static/index.html")):
        return FileResponse(os.path.join(BASE_DIR, "static/index.html"))
    else:
        return None
