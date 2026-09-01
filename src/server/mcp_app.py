"""The shared FastMCP server instance every tool/resource/prompt module
registers against."""

import os
from db.db import init_db,get_conn, put_conn
from mcp.server.fastmcp import FastMCP

port = int(os.environ.get("PORT") or 8001)
mcp = FastMCP("research", host="0.0.0.0", port=port)


@mcp.custom_route("/health", methods=["GET","HEAD"])
async def health(request):
    from starlette.responses import JSONResponse
    db_ok = False
    try:
        conn = get_conn()
        conn.cursor().execute("SELECT 1")
        #put_conn(conn)
        put_conn(conn)
        db_ok = True
    except Exception:
        pass
    return JSONResponse({"status": "ok", "db": db_ok})