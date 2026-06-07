# Copyright (c) 2026 Data-Blitz Inc. All rights reserved.
# License: Proprietary. See NOTICE.md.
# Author: Paul Harvener.

from __future__ import annotations

"""Example MCP client script for exercising Neo4j ontology tools."""

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
    """Connect to local Neo4j ontology MCP server and run a small tool sequence."""

    question = os.getenv(
        "MCP_ONTOLOGY_QUESTION",
        "What does the ontology say about Contract and its related concept?",
    )
    server_script = Path(__file__).resolve().parents[1] / "mcp_servers" / "neo4j_ontology_server.py"
    params = StdioServerParameters(
        command="python",
        args=[str(server_script)],
        env={
            **os.environ,
            "NEO4J_URI": os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            "NEO4J_USER": os.getenv("NEO4J_USER", "neo4j"),
            "NEO4J_PASSWORD": os.getenv("NEO4J_PASSWORD", "password"),
            "NEO4J_DATABASE": os.getenv("NEO4J_DATABASE", "neo4j"),
            "NEO4J_BROWSER_URL": os.getenv("NEO4J_BROWSER_URL", "http://localhost:7474/browser/"),
        },
    )

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()
            tool_names = [tool.name for tool in tools.tools]
            print("Available MCP tools:", ", ".join(tool_names))

            health = await session.call_tool("ontology_health", arguments={})
            print("\nontology_health:")
            print(_extract_text(health))

            context = await session.call_tool(
                "search_ontology_context",
                arguments={"question": question, "node_limit": 8},
            )
            context_text = _extract_text(context)
            print("\nsearch_ontology_context (truncated to 1100 chars):")
            print(context_text[:1100])

            payload = json.loads(context_text) if context_text else {}
            resources = payload.get("resources", [])
            if not resources:
                print("\nNo ontology resources found for the question.")
                return

            first_iri = str(resources[0].get("iri") or "").strip()
            if not first_iri:
                print("\nFirst ontology resource did not include an IRI.")
                return

            detail = await session.call_tool(
                "get_ontology_resource",
                arguments={"iri": first_iri, "neighbor_limit": 10, "literal_limit": 10},
            )
            print("\nget_ontology_resource (truncated to 1100 chars):")
            print(_extract_text(detail)[:1100])

            listing = await session.call_tool(
                "list_ontology_resources",
                arguments={"search": "contract", "limit": 10},
            )
            print("\nlist_ontology_resources:")
            print(_extract_text(listing))


if __name__ == "__main__":
    """Execute demo flow when run as a script."""

    asyncio.run(main())
