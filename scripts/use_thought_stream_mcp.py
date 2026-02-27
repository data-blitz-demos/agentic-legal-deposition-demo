from __future__ import annotations

"""Example MCP client script for exercising thought-stream CouchDB tools."""

import asyncio
import json
import os
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


def _extract_text(result) -> str:
    """Concatenate text fragments from MCP tool call results."""

    parts: list[str] = []
    for item in getattr(result, "content", []):
        text = getattr(item, "text", None)
        if text:
            parts.append(text)
    return "\n".join(parts)


async def main() -> None:
    """Connect to local thought-stream MCP server and run a small tool sequence."""

    trace_id = os.getenv("MCP_THOUGHT_STREAM_TRACE_ID", "demo-trace-001")
    case_id = os.getenv("MCP_THOUGHT_STREAM_CASE_ID", "CASE-001")
    server_script = Path(__file__).resolve().parents[1] / "mcp_servers" / "thought_stream_server.py"
    params = StdioServerParameters(
        command="python",
        args=[str(server_script)],
        env={
            **os.environ,
            "COUCHDB_URL": os.getenv("COUCHDB_URL", "http://admin:password@localhost:5984"),
            "THOUGHT_STREAM_DB": os.getenv("THOUGHT_STREAM_DB", "thought_stream"),
        },
    )

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()
            tool_names = [tool.name for tool in tools.tools]
            print("Available MCP tools:", ", ".join(tool_names))

            health = await session.call_tool("thought_stream_health", arguments={})
            print("\nthought_stream_health:")
            print(_extract_text(health))

            append = await session.call_tool(
                "append_thought_stream_events",
                arguments={
                    "trace_id": trace_id,
                    "case_id": case_id,
                    "status": "running",
                    "legal_clerk": [
                        {
                            "persona": "Persona:Legal Clerk",
                            "phase": "ingest_start",
                            "notes": "Demo thought-stream event",
                        }
                    ],
                },
            )
            print("\nappend_thought_stream_events:")
            print(_extract_text(append))

            detail = await session.call_tool("get_thought_stream", arguments={"trace_id": trace_id})
            detail_text = _extract_text(detail)
            print("\nget_thought_stream (truncated to 900 chars):")
            print(detail_text[:900])

            listed = await session.call_tool("list_thought_streams", arguments={"case_id": case_id, "limit": 10})
            print("\nlist_thought_streams:")
            print(_extract_text(listed))

            payload = json.loads(detail_text) if detail_text else {}
            if payload:
                delete = await session.call_tool("delete_thought_stream", arguments={"trace_id": trace_id})
                print("\ndelete_thought_stream:")
                print(_extract_text(delete))


if __name__ == "__main__":
    """Execute demo flow when run as a script."""

    asyncio.run(main())
