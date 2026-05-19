"""WebSocket query endpoint for the Clinikally SkinAI conversational pipeline.

This module exposes a single ``/ws/query`` WebSocket route that drives the
real-time, streaming interaction between the SkinAI frontend and the
server-side decision-tree agent.  The high-level flow is:

1. The client opens a WebSocket and sends a JSON message containing the user
   query, conversation context, and (optionally) a base-64-encoded skin photo.
2. :func:`process` orchestrates the pipeline: it stores any uploaded image on
   the tree, adjusts routing when a photo is attached, emits an NER
   (Named-Entity Recognition) payload for frontend highlighting, and then
   streams every intermediate and final result back over the socket.
3. On the first message of a new conversation the auto-generated title is
   also sent before the ``completed`` sentinel.
"""

import asyncio
import uuid

from fastapi import APIRouter, Depends, WebSocket
from starlette.websockets import WebSocketDisconnect

from skinai.tree.tree import Tree
from skinai.api.core.log import logger
from skinai.api.dependencies.common import get_user_manager
from skinai.api.services.user import UserManager
from skinai.api.utils.websocket import help_websocket
from skinai.api.utils.ner import named_entity_recognition
from skinai.util.collection import retrieve_all_collection_names
from skinai.api.utils.default_payloads import error_payload

router = APIRouter()


def format_ner_response(text: str, user_id: str, conversation_id: str, query_id: str) -> dict:
    """Build a WebSocket-ready NER payload for the given query text.

    Runs lightweight named-entity recognition over the user's message so
    the frontend can highlight detected skincare entities (ingredients,
    concerns, product types, etc.).

    Args:
        text: Raw user query string.
        user_id: Authenticated user identifier.
        conversation_id: Active conversation identifier.
        query_id: Unique identifier for this query turn.

    Returns:
        A JSON-serialisable dict with ``type: "ner"`` and the NER payload.
    """
    response = named_entity_recognition(text)
    return {
        "type": "ner",
        "id": str(uuid.uuid4()),
        "user_id": user_id,
        "conversation_id": conversation_id,
        "query_id": query_id,
        "payload": response,
    }


async def format_title_response(
    tree: Tree, user_id: str, conversation_id: str, query_id: str
) -> dict:
    """Generate a conversation title and wrap it in a WebSocket payload.

    Called once after the first user prompt so the frontend can label the
    conversation in the sidebar.

    Args:
        tree: The active decision-tree instance.
        user_id: Authenticated user identifier.
        conversation_id: Active conversation identifier.
        query_id: Unique identifier for this query turn.

    Returns:
        A JSON-serialisable dict with ``type: "title"`` containing either the
        generated title or an error string.
    """

    try:
        title = await tree.create_conversation_title_async()
        return {
            "type": "title",
            "id": str(uuid.uuid4()),
            "user_id": user_id,
            "conversation_id": conversation_id,
            "query_id": query_id,
            "payload": {
                "title": title,
                "error": "",
            },
        }
    except Exception as e:
        logger.exception(f"Error in format_title_response")
        return {
            "type": "title",
            "id": str(uuid.uuid4()),
            "user_id": user_id,
            "conversation_id": conversation_id,
            "query_id": query_id,
            "payload": {
                "title": "",
                "error": str(e),
            },
        }


async def process(data: dict, websocket: WebSocket, user_manager: UserManager) -> None:
    """Core query-processing pipeline invoked for each WebSocket message.

    Orchestrates the full lifecycle of a single user turn:

    1. **Image handling** – If ``data["image"]`` contains a base-64 skin photo,
       it is persisted onto the decision tree so that :class:`SkinAnalysisTool`
       can access it.  The query text is also prefixed with a routing hint and
       the ``route`` is force-set to ``"SkinAnalysisTool"``.
    2. **NER** – A named-entity-recognition payload is emitted immediately so
       the frontend can highlight entities while the agent is thinking.
    3. **Streaming** – The decision tree is executed via
       ``user_manager.process_tree()``.  Each yielded result (status updates,
       retrieval objects, text responses) is forwarded over the WebSocket in
       real time.  Internal-only message types (``training_update``, ``timer``)
       are filtered out.
    4. **Title generation** – On the very first turn (``tree_index == 0``) a
       conversation title is auto-generated and sent before the ``completed``
       sentinel.
    5. **Error handling** – Any unhandled exception is caught and sent as an
       ``error_payload`` so the frontend can display a user-friendly message.

    Args:
        data: Incoming JSON message with keys ``user_id``, ``conversation_id``,
            ``query``, ``query_id``, ``route``, ``mimick``,
            ``collection_names``, and optionally ``image``.
        websocket: The active FastAPI WebSocket connection.
        user_manager: Service layer for user / tree lifecycle management.
    """
    logger.debug(f"/query API request received")
    logger.debug(f"User ID: {data['user_id']}")
    logger.debug(f"Conversation ID: {data['conversation_id']}")
    logger.debug(f"Query: {data['query']}")
    logger.debug(f"Query ID: {data['query_id']}")
    logger.debug(f"Training route: {data['route']}")
    logger.debug(f"Training mimick model: {data['mimick']}")
    logger.debug(f"Collection names: {data['collection_names']}")

    user = await user_manager.get_user_local(user_id=data["user_id"])

    # Extract uploaded image base64 if present and assign to tree
    image_data = data.get("image", None)
    if image_data:
        try:
            if user_manager.check_tree_timeout(data["user_id"], data["conversation_id"]):
                if await user_manager.check_tree_exists_weaviate(data["user_id"], data["conversation_id"]):
                    await user_manager.load_tree(data["user_id"], data["conversation_id"])
                else:
                    await user_manager.initialise_tree(data["user_id"], data["conversation_id"])
            
            tree = await user_manager.get_tree(
                user_id=data["user_id"],
                conversation_id=data["conversation_id"],
            )
            tree.last_uploaded_image = image_data
            if hasattr(tree, "tree_data") and tree.tree_data:
                tree.tree_data.last_uploaded_image = image_data
            logger.info("Successfully stored uploaded skin photo in decision tree context.")
        except Exception as ex:
            logger.warning(f"Could not assign uploaded skin photo to tree: {ex}")

    try:
        # optional arguments
        if "route" in data:
            route = data["route"]
        else:
            route = ""

        # When a skin photo is attached, signal the routing LLM and force-route to SkinAnalysisTool
        if image_data:
            if not data["query"].strip():
                data["query"] = "Analyze this skin photo and provide a clinical skincare diagnostic report."
            data["query"] = "[SKIN PHOTO ATTACHED FOR ANALYSIS] " + data["query"]
            route = "SkinAnalysisTool"

        # send ner response in advance
        await websocket.send_json(
            format_ner_response(
                text=data["query"],
                user_id=data["user_id"],
                conversation_id=data["conversation_id"],
                query_id=data["query_id"],
            )
        )

        async for yielded_result in user_manager.process_tree(
            user_id=data["user_id"],
            conversation_id=data["conversation_id"],
            query=data["query"],
            query_id=data["query_id"],
            training_route=route,
            collection_names=data["collection_names"],
        ):
            if asyncio.iscoroutine(yielded_result):
                yielded_result = await yielded_result
            try:
                if (
                    yielded_result is not None
                    and "type" in yielded_result
                    and yielded_result["type"] != "training_update"
                    and yielded_result["type"] != "timer"
                    and yielded_result["type"] != "completed"
                ):
                    await websocket.send_json(yielded_result)

                # before the completed, send title of conversation
                elif (
                    yielded_result is not None
                    and "type" in yielded_result
                    and yielded_result["type"] == "completed"
                ):
                    tree: Tree = await user_manager.get_tree(
                        user_id=data["user_id"],
                        conversation_id=data["conversation_id"],
                    )

                    # only send if it's the first prompt
                    if tree.tree_index == 0:
                        await websocket.send_json(
                            await format_title_response(
                                tree=tree,
                                user_id=data["user_id"],
                                conversation_id=data["conversation_id"],
                                query_id=data["query_id"],
                            )
                        )

                    # send the completed payload
                    await websocket.send_json(yielded_result)

            except WebSocketDisconnect:
                logger.info("Client disconnected during processing")
                break
            # Add a small delay between messages to prevent overwhelming
            await asyncio.sleep(0.005)
            # logger.debug(f"Sent message to client: {yielded_result}")

    except asyncio.CancelledError:
        # In Python 3.9+ CancelledError is a BaseException, not Exception.
        # Weaviate gRPC timeouts surface as CancelledError and must be caught
        # explicitly so the WebSocket stays alive and the client gets feedback.
        logger.warning("Query cancelled (likely Weaviate connection timeout)")
        try:
            error = error_payload(
                text="The database connection timed out. Please try again in a moment.",
                conversation_id=data.get("conversation_id", ""),
                query_id=data.get("query_id", ""),
            )
            await websocket.send_json(error)
            await websocket.send_json({
                "type": "completed",
                "conversation_id": data.get("conversation_id", ""),
                "query_id": data.get("query_id", ""),
            })
        except Exception:
            pass

    except Exception as e:
        logger.exception(f"Error in /query API")

        if "conversation_id" in data:
            error = error_payload(
                text=f"{str(e)}",
                conversation_id=data["conversation_id"],
                query_id=data["query_id"],
            )
            await websocket.send_json(error)
        else:
            error = error_payload(
                text=f"{str(e)}",
                conversation_id="",
                query_id="",
            )
            await websocket.send_json(error)


# Process endpoint
@router.websocket("/query")
async def query_websocket(
    websocket: WebSocket, user_manager: UserManager = Depends(get_user_manager)
) -> None:
    """WebSocket endpoint for the SkinAI conversational query pipeline.

    Accepts a WebSocket upgrade at ``/ws/query``, delegates connection
    lifecycle management (accept / disconnect / keep-alive) to
    :func:`~skinai.api.utils.websocket.help_websocket`, and routes each
    incoming JSON frame through :func:`process`.

    **Expected client JSON schema**::

        {
            "user_id": str,
            "conversation_id": str,
            "query": str,
            "query_id": str,
            "route": str,          // optional forced tool route
            "mimick": str,         // optional training-mode model
            "collection_names": list[str],
            "image": str | null    // optional base-64 skin photo
        }

    **Streamed response types**: ``ner``, ``status``, ``text``, ``result``,
    ``title``, ``error``, ``completed``.
    """

    await help_websocket(websocket, lambda data, ws: process(data, ws, user_manager))
