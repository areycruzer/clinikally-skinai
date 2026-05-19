import asyncio
import json
import requests
import websockets
import sys

def init_user_and_tree(user_id, conv_id):
    print(f"--- Initialising user: {user_id} ---")
    try:
        res = requests.post(f"http://127.0.0.1:8000/init/user/{user_id}", timeout=60)
        print("Init User response status:", res.status_code)
    except Exception as e:
        print("Failed to initialize user:", e)
        sys.exit(1)
        
    print(f"--- Initialising tree for conv: {conv_id} ---")
    try:
        res_tree = requests.post(
            f"http://127.0.0.1:8000/init/tree/{user_id}/{conv_id}",
            json={"low_memory": False},
            timeout=60
        )
        print("Init Tree response status:", res_tree.status_code)
    except Exception as e:
        print("Failed to initialize tree:", e)
        sys.exit(1)

async def run_query(user_id, conv_id, query_text):
    uri = "ws://127.0.0.1:8000/ws/query"
    print(f"\n==========================================")
    print(f"RUNNING QUERY: \"{query_text}\"")
    print(f"==========================================")
    
    async with websockets.connect(uri) as websocket:
        payload = {
            "user_id": user_id,
            "conversation_id": conv_id,
            "query": query_text,
            "query_id": f"q_{hash(query_text) % 100000}",
            "route": "",
            "mimick": False,
            "collection_names": ["SkincareProducts", "SkincareBlogs"]
        }
        await websocket.send(json.dumps(payload))
        
        full_response = []
        selected_tools = []
        
        while True:
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=120.0)
                resp_json = json.loads(response)
                msg_type = resp_json.get("type")
                
                if msg_type == "error":
                    print("❌ Error response:", resp_json)
                    break
                elif msg_type == "completed":
                    print("\n🎉 Processing completed!")
                    break
                elif msg_type == "message":
                    payload_data = resp_json.get("payload", {})
                    if isinstance(payload_data, dict):
                        text = payload_data.get("text", "")
                        print(text, end="", flush=True)
                        full_response.append(text)
                    else:
                        print(payload_data)
                elif msg_type == "decision":
                    # For decision messages from the server
                    payload_data = resp_json.get("payload", {})
                    node = payload_data.get('node')
                    decision = payload_data.get('decision')
                    reasoning = payload_data.get('reasoning')
                    print(f"\n[Decision Node: {node} -> Selected Action: {decision}]")
                    print(f"[Reasoning: {reasoning}]")
                    if decision:
                        selected_tools.append(decision)
                elif msg_type == "tree_update":
                    # Tree updates also tell us decisions
                    payload_data = resp_json.get("payload", {})
                    node = payload_data.get('node')
                    decision = payload_data.get('decision')
                    reasoning = payload_data.get('reasoning')
                    print(f"\n[Tree Decision on: {node} -> {decision}]")
                    if decision and decision not in selected_tools:
                        selected_tools.append(decision)
                elif msg_type == "status":
                    payload_data = resp_json.get("payload", {})
                    print(f"\n* Status: {payload_data.get('text')} *")
            except asyncio.TimeoutError:
                print("\n⏳ Timeout reached while waiting for response.")
                break
            except Exception as e:
                print(f"\nException in receive loop: {e}")
                break
                
        return "".join(full_response), selected_tools

async def main():
    # We will use clean sessions or the same session to verify. Let's do distinct conversation IDs to avoid mixing states for clean tests.
    queries = [
        ("moisturiser_query", "Recommend a moisturiser under ₹1200 for oily skin"),
        ("niacinamide_query", "What does niacinamide do for the skin?"),
        ("acne_query", "How do I treat hormonal acne?")
    ]
    
    results = {}
    for conv_id, q_text in queries:
        init_user_and_tree("clinikally_user", conv_id)
        resp, tools = await run_query("clinikally_user", conv_id, q_text)
        results[q_text] = {
            "response": resp,
            "tools_used": tools
        }
        print("\n\n")
        
    print("==========================================")
    print("SUMMARY OF TEST RESULTS")
    print("==========================================")
    for q, data in results.items():
        print(f"Query: {q}")
        print(f"Tools Used: {data['tools_used']}")
        print(f"Response: {data['response'][:200]}...")
        print("-" * 40)

if __name__ == "__main__":
    asyncio.run(main())
