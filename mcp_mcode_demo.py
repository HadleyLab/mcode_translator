#!/usr/bin/env python3
# mcp_mcode_fact_demo.py
# Store & verify visible content in mCODE (Python-Script source).
import os, uuid, json, time, argparse, requests
from dotenv import load_dotenv

MCP_URL = "https://core.heysol.ai/api/v1/mcp?source=Python-Script"
SESSION_ID = None  # captured on initialize()

def _headers():
    h = {
        "Authorization": f"Bearer {os.environ['COREAI_API_KEY']}",
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream, */*",
    }
    if SESSION_ID:
        h["Mcp-Session-Id"] = SESSION_ID
    return h

def _rpc(payload, stream=False, debug=False):
    if debug:
        print("\n>> POST", MCP_URL)
        print(">> PAYLOAD:", json.dumps(payload, indent=2, ensure_ascii=False))
    r = requests.post(MCP_URL, json=payload, headers=_headers(), timeout=60, stream=stream)
    try:
        r.raise_for_status()
    except requests.HTTPError:
        print("!! Server said:", r.text)
        raise
    ctype = (r.headers.get("Content-Type") or "").split(";")[0].strip()
    if ctype == "application/json":
        msg = r.json()
    elif ctype == "text/event-stream":
        msg = None
        for line in r.iter_lines(decode_unicode=True):
            if line.startswith("data:"):
                msg = json.loads(line[5:].strip()); break
        if msg is None:
            raise RuntimeError("No JSON in SSE stream")
    else:
        raise RuntimeError(f"Unexpected Content-Type: {ctype}")
    if "error" in msg:
        raise RuntimeError(json.dumps(msg["error"], ensure_ascii=False))
    return r, msg["result"]

def _unwrap(result):
    # Many tools reply with {"content":[{"type":"text","text":"{...json...}"}]}
    if isinstance(result, dict) and "content" in result and isinstance(result["content"], list):
        for item in result["content"]:
            if item.get("type") == "text":
                txt = item.get("text", "")
                try: return json.loads(txt)
                except json.JSONDecodeError: return txt
    return result

def initialize(debug=False):
    global SESSION_ID
    r, out = _rpc({
        "jsonrpc":"2.0","id":str(uuid.uuid4()),
        "method":"initialize",
        "params":{"protocolVersion":"1.0.0","capabilities":{"tools":True},
                  "clientInfo":{"name":"mCODE-VisibleFacts","version":"0.3.0"}}
    }, stream=False, debug=debug)
    SESSION_ID = r.headers.get("Mcp-Session-Id") or SESSION_ID
    if debug: print("Mcp-Session-Id:", SESSION_ID)
    return out

def tools_list(debug=False):
    _, out = _rpc({"jsonrpc":"2.0","id":str(uuid.uuid4()),"method":"tools/list","params":{}},
                  stream=False, debug=debug)
    return out

def call_tool(name, arguments, stream=True, debug=False):
    _, raw = _rpc({
        "jsonrpc":"2.0","id":str(uuid.uuid4()),
        "method":"tools/call",
        "params":{"name":name,"arguments":arguments}
    }, stream=stream, debug=debug)
    if debug:
        print("<< RAW TOOL RESULT:", json.dumps(raw, indent=2, ensure_ascii=False))
    return _unwrap(raw)

def get_space_id_by_name(space_name, debug=False):
    raw = call_tool("memory_get_spaces", {}, stream=False, debug=debug)
    spaces = []
    if isinstance(raw, dict) and "spaces" in raw:
        spaces = raw["spaces"]
    elif isinstance(raw, list):
        spaces = raw
    else:
        # content-wrapped array as string
        try:
            spaces = json.loads(raw)
        except Exception:
            spaces = []
    if debug:
        print("Spaces:", json.dumps(spaces, indent=2, ensure_ascii=False))
    for s in spaces:
        if s.get("name") == space_name and s.get("writable", True):
            return s.get("id") or s.get("spaceId")
    raise RuntimeError(f"Space '{space_name}' not found or not writable")

def get_ingest_schema(debug=False):
    tl = tools_list(debug=debug)
    for t in tl.get("tools", []):
        if t["name"] == "memory_ingest":
            return t.get("inputSchema") or {}
    raise RuntimeError("memory_ingest not available")

def ingest_fact_visible(statement, space_id, debug=False):
    """
    Send statement + common display fields WHEN allowed by schema.
    This helps UIs that render title/summary show non-blank cards.
    """
    schema = get_ingest_schema(debug=debug)
    props = schema.get("properties", {}) if isinstance(schema.get("properties"), dict) else {}
    required = set(schema.get("required", [])) if isinstance(schema.get("required"), list) else set()

    args = {"statement": statement}
    if "spaceId" in props or "spaceId" in required:
        args["spaceId"] = space_id

    # Add display fields if the schema advertises them
    if "title" in props:
        args["title"] = statement[:80]
    if "summary" in props:
        # short summary for card UIs
        args["summary"] = statement
    if "source" in props:
        args["source"] = "Python-Script"
    if "tags" in props:
        args["tags"] = ["mCODE","fact"]
    if "metadata" in props:
        args["metadata"] = {"via":"mcp_mcode_fact_demo.py","len":len(statement)}

    if debug:
        print("Ingest FACT args:", json.dumps(args, indent=2, ensure_ascii=False))
    return call_tool("memory_ingest", args, stream=True, debug=debug)

def ingest_episode_optional(text, space_id, debug=False):
    """
    Some UIs show episodes more prominently. This is optional; skip if not wanted.
    """
    schema = get_ingest_schema(debug=debug)
    props = schema.get("properties", {}) if isinstance(schema.get("properties"), dict) else {}
    if "episodeBody" not in props:
        return {"skipped": "episodeBody not supported by schema"}
    args = {"episodeBody": text}
    if "spaceId" in props:
        args["spaceId"] = space_id
    if "title" in props:
        args["title"] = text[:80]
    if "metadata" in props:
        args["metadata"] = {"via":"mcp_mcode_fact_demo.py","type":"episode"}
    if debug:
        print("Ingest EPISODE args:", json.dumps(args, indent=2, ensure_ascii=False))
    return call_tool("memory_ingest", args, stream=True, debug=debug)

def search_facts(query, space_id, limit=5, debug=False):
    # Many servers accept spaceId for scoped search; send it.
    args = {"query": query, "limit": limit, "spaceId": space_id}
    res = call_tool("memory_search", args, stream=False, debug=debug)
    return res  # typically {"episodes":[...], "facts":[...]}

def main():
    load_dotenv()
    if "COREAI_API_KEY" not in os.environ:
        raise RuntimeError("Put COREAI_API_KEY in your environment or a .env file")

    ap = argparse.ArgumentParser(description="Store a visible FACT in mCODE and verify via facts search")
    ap.add_argument("--space", default="mCODE")
    ap.add_argument("--text", default="NCT00616135 is a Phase II breast cancer trial at AdventHealth")
    ap.add_argument("--query", default="NCT00616135")
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--also-episode", action="store_true", help="Also write an episodeBody entry")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    initialize(debug=args.debug)
    space_id = get_space_id_by_name(args.space, debug=args.debug)

    # 1) FACT (with display fields if supported)
    fact_ack = ingest_fact_visible(args.text, space_id, debug=args.debug)
    print("FACT ingest ACK:", json.dumps(fact_ack, indent=2, ensure_ascii=False))

    # 2) (optional) EPISODE
    if args.also_episode:
        ep_ack = ingest_episode_optional(args.text, space_id, debug=args.debug)
        print("EPISODE ingest ACK:", json.dumps(ep_ack, indent=2, ensure_ascii=False))

    # 3) Verify: search facts (and show full objects)
    time.sleep(0.6)  # tiny delay in case of eventual indexing
    found = search_facts(args.query, space_id, limit=args.limit, debug=args.debug)
    print("Search result (full):", json.dumps(found, indent=2, ensure_ascii=False))

    # Convenience: print just the facts array if present
    if isinstance(found, dict) and "facts" in found:
        print("\nFacts only:")
        print(json.dumps(found["facts"], indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
