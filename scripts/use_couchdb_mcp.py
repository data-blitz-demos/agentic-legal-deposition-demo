from __future__ import annotations

"""Example MCP client script for exercising CouchDB deposition tools."""

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
    """Connect to local MCP server script and run a small tool sequence."""

    case_id = os.getenv("MCP_DEMO_CASE_ID", "CASE-001")
    server_script = Path(__file__).resolve().parents[1] / "mcp_servers" / "couchdb_server.py"
    params = StdioServerParameters(
        command="python",
        args=[str(server_script)],
        env={
            **os.environ,
            "COUCHDB_URL": os.getenv("COUCHDB_URL", "http://admin:password@localhost:5984"),
            "COUCHDB_DB": os.getenv("COUCHDB_DB", "depositions"),
        },
    )

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()
            tool_names = [tool.name for tool in tools.tools]
            print("Available MCP tools:", ", ".join(tool_names))

            listed = await session.call_tool(
                "list_case_depositions",
                arguments={"case_id": case_id, "limit": 10},
            )
            listed_text = _extract_text(listed)
            print("\nlist_case_depositions result:")
            print(listed_text)

            payload = json.loads(listed_text) if listed_text else {}
            depositions = payload.get("depositions", [])
            if not depositions:
                print("\nNo depositions found. Ingest case data first via UI or /api/ingest-case.")
                return

            first_id = depositions[0].get("_id")
            if not first_id:
                print("\nFirst deposition missing _id.")
                return

            detail = await session.call_tool("get_deposition", arguments={"deposition_id": first_id})
            print("\nget_deposition result (truncated to 900 chars):")
            detail_text = _extract_text(detail)
            print(detail_text[:900])


if __name__ == "__main__":
    """Execute demo flow when run as a script."""

    asyncio.run(main())
