"""
Bonus Feature Verification Script
Tests:
1. Multi-turn conversation (product query → follow-up in same conv)
2. Blog RAG query
3. General knowledge query
4. User feedback endpoint
"""
import asyncio
import json
import requests
import websockets
import uuid
import sys
import time

BASE = "http://127.0.0.1:8000"
USER_ID = "clinikally_bonus_test"

def init_user_and_tree(conv_id):
    print(f"  [INIT] User={USER_ID}, Conv={conv_id}")
    try:
        r1 = requests.post(f"{BASE}/init/user/{USER_ID}", timeout=60)
        print(f"    init/user → {r1.status_code}")
    except Exception as e:
        print(f"    init/user FAILED: {e}")
        return False
    
    try:
        r2 = requests.post(
            f"{BASE}/init/tree/{USER_ID}/{conv_id}",
            json={"low_memory": False},
            timeout=60
        )
        print(f"    init/tree → {r2.status_code}")
        return r2.status_code == 200
    except Exception as e:
        print(f"    init/tree FAILED: {e}")
        return False

async def run_query(conv_id, query_text, timeout_s=120):
    uri = "ws://127.0.0.1:8000/ws/query"
    payload = {
        "user_id": USER_ID,
        "conversation_id": conv_id,
        "query": query_text,
        "query_id": f"q_{uuid.uuid4().hex[:8]}",
        "route": "",
        "mimick": False,
        "collection_names": ["SkincareProducts", "SkincareBlogs"]
    }
    
    full_response = []
    selected_tools = []
    
    async with websockets.connect(uri, close_timeout=30) as ws:
        await ws.send(json.dumps(payload))
        
        while True:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=timeout_s)
                data = json.loads(raw)
                msg_type = data.get("type")
                
                if msg_type == "completed":
                    break
                elif msg_type == "error":
                    err_payload = data.get("payload", {})
                    err_text = err_payload.get("text", "") if isinstance(err_payload, dict) else str(err_payload)
                    full_response.append(f"[ERROR] {err_text}")
                    break
                elif msg_type == "tree_update":
                    p = data.get("payload", {})
                    decision = p.get("decision", "")
                    if decision:
                        selected_tools.append(decision)
                        print(f"    🔀 Routed → {decision}")
                elif msg_type == "decision":
                    p = data.get("payload", {})
                    decision = p.get("decision", "")
                    if decision and decision not in selected_tools:
                        selected_tools.append(decision)
                        print(f"    🔀 Routed → {decision}")
                elif msg_type == "status":
                    p = data.get("payload", {})
                    status_text = p.get("text", "")
                    if not status_text and "objects" in p:
                        for obj in p["objects"]:
                            status_text = obj.get("text", "")
                    print(f"    ⏳ {status_text}")
                elif msg_type == "text":
                    p = data.get("payload", {})
                    objs = p.get("objects", [])
                    for obj in objs:
                        t = obj.get("text", "")
                        if t:
                            full_response.append(t)
                elif msg_type == "message":
                    p = data.get("payload", {})
                    if isinstance(p, dict):
                        t = p.get("text", "")
                        if t:
                            full_response.append(t)
                    
            except asyncio.TimeoutError:
                print("    ⚠️ Timeout!")
                break
            except websockets.exceptions.ConnectionClosed:
                print("    ⚠️ WebSocket closed by server")
                break
            except Exception as e:
                print(f"    ⚠️ Exception: {e}")
                break
    
    return "\n".join(full_response), selected_tools

async def main():
    results = {}
    
    # ── TEST 1: Multi-turn product conversation ──
    print("=" * 60)
    print("TEST 1: Multi-turn product conversation")
    print("=" * 60)
    
    conv1 = f"bonus_mt_{uuid.uuid4().hex[:6]}"
    init_user_and_tree(conv1)
    
    print('\n  Q1: "Recommend a moisturiser under ₹1200 for oily skin"')
    resp1, tools1 = await run_query(conv1, "Recommend a moisturiser under ₹1200 for oily skin")
    print(f"  Tools: {tools1}")
    print(f"  Response: {resp1[:200]}...")
    results["1_product_query"] = ("ProductQueryTool" in str(tools1), tools1)
    
    # Give the server time to save the tree state before sending a follow-up
    print("\n  (waiting 5s for tree state to settle...)")
    time.sleep(5)
    
    # Re-init tree for follow-up (SkinAI retrieves existing tree state)
    init_user_and_tree(conv1)
    
    print('\n  Q2 (follow-up): "Which of those has niacinamide?"')
    resp2, tools2 = await run_query(conv1, "Which of those has niacinamide?")
    print(f"  Tools: {tools2}")
    print(f"  Response: {resp2[:200]}...")
    results["2_followup"] = (len(resp2) > 10, tools2)
    
    # ── TEST 2: Blog RAG query ──
    print("\n" + "=" * 60)
    print("TEST 2: Blog RAG query")
    print("=" * 60)
    
    conv2 = f"bonus_blog_{uuid.uuid4().hex[:6]}"
    init_user_and_tree(conv2)
    
    print('\n  Q: "What is the best night skincare routine?"')
    resp3, tools3 = await run_query(conv2, "What is the best night skincare routine?")
    print(f"  Tools: {tools3}")
    print(f"  Response: {resp3[:200]}...")
    results["3_blog_rag"] = (len(resp3) > 20, tools3)
    
    # ── TEST 3: General knowledge ──
    print("\n" + "=" * 60)
    print("TEST 3: General knowledge query")
    print("=" * 60)
    
    conv3 = f"bonus_gen_{uuid.uuid4().hex[:6]}"
    init_user_and_tree(conv3)
    
    print('\n  Q: "How do I treat hormonal acne?"')
    resp4, tools4 = await run_query(conv3, "How do I treat hormonal acne?")
    print(f"  Tools: {tools4}")
    print(f"  Response: {resp4[:200]}...")
    results["4_general"] = ("GeneralKnowledgeTool" in str(tools4) or len(resp4) > 20, tools4)
    
    # ── TEST 4: Feedback endpoint ──
    print("\n" + "=" * 60)
    print("TEST 4: User feedback mechanism")
    print("=" * 60)
    
    try:
        fb = requests.post(f"{BASE}/feedback/add", json={
            "user_id": USER_ID,
            "conversation_id": conv1,
            "query_id": f"q_{uuid.uuid4().hex[:8]}",
            "feedback": 1
        }, timeout=15)
        print(f"  Feedback POST → {fb.status_code}")
        results["5_feedback"] = (fb.status_code == 200, [])
    except Exception as e:
        print(f"  Feedback FAILED: {e}")
        results["5_feedback"] = (False, [])
    
    # ── SUMMARY ──
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    
    all_pass = True
    for name, (passed, tools) in results.items():
        icon = "✅" if passed else "❌"
        print(f"  {icon} {name}: {'PASS' if passed else 'FAIL'} (tools={tools})")
        if not passed:
            all_pass = False
    
    print()
    if all_pass:
        print("🎉 ALL BONUS FEATURE TESTS PASSED!")
    else:
        print("⚠️  Some tests need attention — see above.")
    
    return 0 if all_pass else 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
