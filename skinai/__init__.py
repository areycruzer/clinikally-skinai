"""Clinikally SkinAI – AI-powered clinical skincare assistant.

This is the top-level package for the SkinAI platform.  It re-exports the
core building blocks needed to build and extend the conversational agent
pipeline:

* :class:`Tree` – the decision-tree orchestrator that routes user queries to
  the appropriate tool.
* **Return types** (``Response``, ``Result``, ``Retrieval``, ``Error``, …) –
  objects yielded by tools and streamed to the frontend over WebSocket.
* :class:`Tool` / :func:`tool` – base class and decorator for defining custom
  agent tools.
* **Preprocessing helpers** – utilities for creating and managing Weaviate
  collections from raw data.
* :class:`Settings` / :func:`configure` – centralised configuration and
  environment setup.
"""

from skinai.__metadata__ import (
    __version__,
    __name__,
    __description__,
    __url__,
    __author__,
    __author_email__,
)

from skinai.tree.tree import Tree
from skinai.objects import (
    Tool,
    Return,
    Text,
    Response,
    Update,
    Status,
    Warning,
    Error,
    Completed,
    Result,
    Retrieval,
    tool,
)
from skinai.preprocessing.collection import (
    preprocess,
    preprocessed_collection_exists,
    edit_preprocessed_collection,
    delete_preprocessed_collection,
    view_preprocessed_collection,
)
from skinai.config import Settings, settings, configure, smart_setup, set_from_env
