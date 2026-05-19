# Clinikally SkinAI — API Reference

This document describes the backend API endpoints for the Clinikally SkinAI agentic skincare assistant.

---

## Base URL

```
http://localhost:8000
```

---

## REST Endpoints

### 1. Initialize User

Creates or loads a user session. Must be called before any queries.

```
POST /init/user/{user_id}
```

| Parameter | Type | Location | Description |
|-----------|------|----------|-------------|
| `user_id` | string | URL path | Unique user identifier |

**Response** `200 OK`:
```json
{ "error": "" }
```

**Example:**
```bash
curl -X POST http://localhost:8000/init/user/user123
```

---

### 2. Initialize Conversation Tree

Creates or loads a conversation decision tree for the given user/conversation pair.

```
POST /init/tree/{user_id}/{conversation_id}
```

| Parameter | Type | Location | Description |
|-----------|------|----------|-------------|
| `user_id` | string | URL path | User identifier |
| `conversation_id` | string | URL path | Conversation identifier |
| `low_memory` | boolean | JSON body (optional) | Use low-memory mode (default: `false`) |

**Response** `200 OK`:
```json
{ "error": "" }
```

**Example:**
```bash
curl -X POST http://localhost:8000/init/tree/user123/conv_abc \
  -H "Content-Type: application/json" \
  -d '{"low_memory": false}'
```

---

### 3. Submit Feedback

Records user feedback (👍 thumbs up / 👎 thumbs down) on a specific response.

```
POST /feedback/add
```

**Request body:**
```json
{
  "user_id": "user123",
  "conversation_id": "conv_abc",
  "query_id": "q_001",
  "feedback": 1
}
```

| Field | Type | Description |
|-------|------|-------------|
| `user_id` | string | User identifier |
| `conversation_id` | string | Conversation identifier |
| `query_id` | string | Query ID that the feedback applies to |
| `feedback` | integer | `1` = helpful (👍), `-1` = not helpful (👎) |

**Response** `200 OK`:
```json
{ "error": "" }
```

---

### 4. Remove Feedback

Removes previously submitted feedback.

```
POST /feedback/remove
```

**Request body:**
```json
{
  "user_id": "user123",
  "conversation_id": "conv_abc",
  "query_id": "q_001"
}
```

---

### 5. Health Check

Liveness probe for container orchestrators and uptime monitors.

```
GET /api/health
```

**Response** `200 OK`:
```json
{ "status": "healthy" }
```

---

## WebSocket Endpoint

### Main Chat — `/ws/query`

The primary conversational endpoint. Opens a bidirectional WebSocket connection for real-time streaming interaction with the SkinAI agent.

#### Connection Flow

```
1. Client → POST /init/user/{user_id}
2. Client → POST /init/tree/{user_id}/{conversation_id}
3. Client → WebSocket connect to ws://host:8000/ws/query
4. Client → Send JSON query message
5. Server → Stream multiple response frames (status, text, tree_update, decision, completed)
6. Connection closes after "completed" frame
```

#### Client → Server (Query Message)

```json
{
  "user_id": "user123",
  "conversation_id": "conv_abc",
  "query": "Recommend a moisturiser under ₹1200 for oily skin",
  "query_id": "q_001",
  "route": "",
  "mimick": false,
  "collection_names": ["SkincareProducts", "SkincareBlogs"],
  "image": "data:image/jpeg;base64,/9j/4AAQ..."
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `user_id` | string | ✅ | User identifier |
| `conversation_id` | string | ✅ | Conversation identifier |
| `query` | string | ✅ | User's natural-language query |
| `query_id` | string | ✅ | Unique ID for this query (used for feedback) |
| `route` | string | ❌ | Force a specific tool (e.g., `"ProductQueryTool"`) |
| `mimick` | boolean | ❌ | Mimick mode flag (default: `false`) |
| `collection_names` | string[] | ✅ | Weaviate collections to search |
| `image` | string | ❌ | Base64-encoded image data URL for skin photo analysis |

#### Server → Client (Response Frames)

The server streams multiple JSON frames. Each frame has a `type` field:

| Type | Description | Payload |
|------|-------------|---------|
| `ner` | Named entity recognition results for frontend highlighting | `{ "objects": [...] }` |
| `status` | Agent status update (e.g., "Querying skincare products...") | `{ "text": "..." }` |
| `tree_update` | Decision tree routing update | `{ "decision": "ProductQueryTool", "reasoning": "..." }` |
| `decision` | Final tool decision | `{ "decision": "BlogRAGTool" }` |
| `text` | Streamed response text chunk | `{ "objects": [{ "text": "..." }] }` |
| `message` | Alternative text format | `{ "text": "..." }` |
| `object` | Retrieved Weaviate objects (product cards) | `{ "objects": [...] }` |
| `error` | Error notification | `{ "text": "Error message" }` |
| `completed` | End-of-response sentinel | `{}` |

#### Example with `wscat`

```bash
# Install wscat: npm install -g wscat
wscat -c ws://localhost:8000/ws/query

# Send query:
{"user_id":"test","conversation_id":"conv1","query":"What does niacinamide do?","query_id":"q1","route":"","mimick":false,"collection_names":["SkincareProducts","SkincareBlogs"]}
```

#### Image Upload Flow

When a base64-encoded image is included in the `image` field:

1. The image is stored on the decision tree's state (`tree.last_uploaded_image`)
2. The query text is prefixed with `[SKIN PHOTO ATTACHED FOR ANALYSIS]`
3. The route is force-set to `SkinAnalysisTool`
4. The tool sends the image to Gemini Vision (with multi-model retry chain)
5. A clinical diagnostic report is streamed back

---

## Error Handling

All REST endpoints return errors in a consistent format:

```json
{ "error": "Description of the error" }
```

HTTP status codes:
- `200` — Success
- `400` — Bad request (missing or invalid parameters)
- `500` — Internal server error

The WebSocket endpoint sends error frames with `type: "error"` and gracefully closes the connection. The agent also implements graceful degradation: if the database is unreachable, it falls back to the GeneralKnowledgeTool to ensure the user always receives a response.

---

## Authentication

The current implementation uses user IDs for session management without formal authentication. For production deployment, JWT or OAuth2 authentication should be added as middleware.
