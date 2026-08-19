"""The shared FastMCP server instance every tool/resource/prompt module
registers against."""

import os
from mcp.server.fastmcp import FastMCP

port = int(os.environ.get("PORT") or 8001)
mcp = FastMCP("research", host="0.0.0.0", port=port)