"""Tavily Remote MCP: one client, its tools shared by several agents, every call traced.

The tools are discovered at runtime from the remote server (Streamable HTTP transport,
API key as a URL query parameter). Research and Code both call `tavily_search` in
different contexts, Summary calls `tavily_extract`. Each call emits an `mcp_tool_call`
trace event naming the calling agent, so the sharing is visible in the UI.
"""

import asyncio
import json
import time

from src.config import get_settings
from src.trace import emit

SERVER = "tavily"


class TavilyMCP:
    def __init__(self):
        self._tools: dict | None = None
        self._lock = asyncio.Lock()

    async def tools(self) -> dict:
        if self._tools is None:
            async with self._lock:
                if self._tools is None:
                    from langchain_mcp_adapters.client import MultiServerMCPClient

                    client = MultiServerMCPClient(
                        {SERVER: {"transport": "streamable_http", "url": get_settings().tavily_mcp_url, "timeout": 30}})
                    self._tools = {t.name: t for t in await client.get_tools()}
        return self._tools

    async def call(self, tool_name: str, args: dict, *, agent: str) -> dict:
        t0 = time.perf_counter()
        try:
            tool = (await self.tools())[tool_name]
            blocks = await tool.ainvoke(args)
            text = blocks[0]["text"] if isinstance(blocks, list) and blocks else str(blocks)
            payload = json.loads(text) if text.lstrip().startswith("{") else {"results": [], "raw": text}
        except Exception as e:
            emit("mcp_tool_call", {"agent": agent, "server": SERVER, "tool": tool_name, "args": args,
                                   "status": "error", "error": repr(e)[:200],
                                   "latency_s": round(time.perf_counter() - t0, 3)})
            raise
        emit("mcp_tool_call", {"agent": agent, "server": SERVER, "tool": tool_name, "args": args, "status": "ok",
                               "n_results": len(payload.get("results", [])),
                               "latency_s": round(time.perf_counter() - t0, 3)})
        return payload


tavily = TavilyMCP()
