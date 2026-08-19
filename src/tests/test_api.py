"""
test_api.py — Quick CLI test for the FastAPI SSE endpoints.
Run AFTER starting: uv run uvicorn api:app --reload --port 8000

Usage:
  uv run python test_api.py
"""

import httpx
import json
import asyncio




#API_BASE = "http://localhost:8000"
API_BASE = "https://ragchatbot-api-0sf4.onrender.com"

async def stream_chat(query: str, session_id: str | None = None):
    """Send a query and print SSE events as they arrive."""
    print(f"\n{'='*60}")
    print(f"QUERY: {query}")
    print(f"{'='*60}")

    current_session = session_id

    async with httpx.AsyncClient(timeout=120) as client:
        async with client.stream(
            "POST",
            f"{API_BASE}/chat",
            json={"query": query, "session_id": session_id},
        ) as response:
            async for line in response.aiter_lines():
                if not line:
                    continue

                if line.startswith("event:"):
                    event_type = line[6:].strip()
                elif line.startswith("data:"):
                    data = json.loads(line[5:].strip())

                    if event_type == "tool_start":
                        print(f"🔧 Tool: {data['tool']} | input: {data['input']}")

                    elif event_type == "tool_end":
                        print(f"✅ Tool done: {data['tool']}")

                    elif event_type == "token":
                        print(data["content"], end="", flush=True)

                    elif event_type == "interrupt":
                        print(f"\n\n❓ CLARIFICATION NEEDED: {data['question']}")
                        if data.get("options"):
                            print(f"   Options: {', '.join(data['options'])}")
                        current_session = data["session_id"]
                        answer = input("\nYour answer: ").strip()
                        await stream_resume(current_session, answer)
                        return

                    elif event_type == "done":
                        print(f"\n\n✅ DONE call in stream_chat (session: {data['session_id']})")
                        current_session = data["session_id"]
                        print(f"\n\nANSWER from stream_chat: {data.get('answer', '(empty)')}")
                       
                    elif event_type == "error":
                        print(f"\n❌ ERROR: {data['message']}")


async def stream_resume(session_id: str, answer: str):
    """Resume after interrupt."""
    print(f"\n{'='*60}")
    print(f"RESUMING with: {answer}")
    print(f"{'='*60}")

    async with httpx.AsyncClient(timeout=120) as client:
        async with client.stream(
            "POST",
            f"{API_BASE}/resume",
            json={"session_id": session_id, "answer": answer},
        ) as response:
            async for line in response.aiter_lines():
                if not line:
                    continue
                if line.startswith("event:"):
                    event_type = line[6:].strip()
                elif line.startswith("data:"):
                    data = json.loads(line[5:].strip())
                    if event_type == "tool_start":
                        print(f"🔧 Tool: {data['tool']}")
                    elif event_type == "token":
                        print(data["content"], end="", flush=True)
                    elif event_type == "done":
                        print(f"\n\n✅ DONE from stream_resume ")
                        print(f"\n\nANSWER from stream_resume : {data.get('answer', '(empty)')}")
                    elif event_type == "error":
                        print(f"\n❌ ERROR: {data['message']}")


async def main():
    # Test 1: clear query (no interrupt expected)
    await stream_chat("search papers on large language models")

    # Test 2: ambiguous query (interrupt expected)
    await stream_chat("search papers on POPE")


if __name__ == "__main__":
    asyncio.run(main())