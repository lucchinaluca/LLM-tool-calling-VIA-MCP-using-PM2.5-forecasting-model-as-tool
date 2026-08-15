import os
os.environ["MCP_USE_ANONYMIZED_TELEMETRY"] = "false"
os.environ["DEBUG"] = "0"
import requests
from datetime import datetime, timezone
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("bath-pollution-mcp")

@mcp.tool()
def get_pollution_bath() -> dict:
    """Return latest pollution data for Bath using OpenAQ."""
    url = "https://api.openaq.org/v2/latest"
    params = {"city": "Bath"}

    r = requests.get(url, params=params).json()

    if "results" not in r or len(r["results"]) == 0:
        return {
            "tool_name": "get_pollution_bath",
            "error": "No pollution data available",
            "source": "OpenAQ latest endpoint",
            "location": "Bath, UK",
            "fetched_at": datetime.now(timezone.utc).isoformat()
        }

    measurements = r["results"][0]["measurements"]
    values = {m["parameter"]: m["value"] for m in measurements}

    return {
        "tool_name": "get_pollution_bath",
        "source": "OpenAQ latest endpoint",
        "location": "Bath, UK",
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "summary": f"Latest pollution values for Bath, returned {len(values)} parameters.",
        "measurements": values,
    }

if __name__ == "__main__":
    mcp.run(transport="stdio")
