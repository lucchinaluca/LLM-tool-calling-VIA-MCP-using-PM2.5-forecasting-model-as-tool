import asyncio
import json
import re
import sys
from mcp import ClientSessionGroup, StdioServerParameters

from openai import OpenAI

DEFAULT_MODEL = "qwen2.5:7b"
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="not-needed"
)


def build_system_prompt() -> str:
    return (
        "You are an assistant with access to real-time tools for Bath weather, Bath pollution, and PM2.5 forecasting.\n"
        "IMPORTANT: You must ALWAYS call the appropriate tool(s) when the user asks about weather, air quality, "
        "pollution, PM2.5 levels, forecasts, or any outdoor activity planning. "
        "Never answer from memory or make up data — always use the tools.\n"
        "Available tools:\n"
        "- get_weather_bath: fetches real-time Bath weather and a 12-hour hourly forecast.\n"
        "- get_pollution_bath: fetches current Bath pollution measurements.\n"
        "- predict_bath_pm25_forecast: predicts the next 24 hours of Bath PM2.5 using a Transformer-LSTM model.\n"
        "If the user asks about current and future conditions, call ALL relevant tools before answering."
    )


def is_tool_relevant(prompt: str) -> bool:
    return bool(re.search(
        r"\b(weather|pollution|air quality|PM2\.5|forecast|forecasting|run|jog|tomorrow|plan|outdoor|outside|conditions|exercise|walk|breathe|smog)\b",
        prompt,
        re.IGNORECASE
    ))


def get_tool_description(tool_name: str, default: str) -> str:
    descriptions = {
        "get_weather_bath": (
            "Fetch real-time Bath weather from Open-Meteo and return a 12-hour hourly forecast. "
            "Includes temperature, wind speed, wind gusts, rain, precipitation, apparent temperature, and dew point."
        ),
        "get_pollution_bath": (
            "Fetch the latest Bath pollution measurements from OpenAQ. "
            "Returns current values for pollutants such as PM2.5, PM10, NO2, O3, etc. "
            "Use this to assess whether outdoor exercise is safe."
        ),
        "predict_bath_pm25_forecast": (
            "Predict the next 1–24 hours of Bath PM2.5 concentration using a saved Transformer-LSTM model. "
            "Uses the last 72 hours of live PM2.5 data from Open-Meteo as model input. "
            "Accepts an optional `hours_to_forecast` integer parameter (default 24, max 24)."
        ),
    }
    return descriptions.get(tool_name, default)


def sanitize_input_schema(raw_schema: dict) -> dict:
    """
    Strip top-level fields that confuse Ollama's OpenAI shim (title, $schema, etc.)
    and ensure the schema always has at least an empty properties object.
    Ollama requires: {"type": "object", "properties": {...}, "required": [...]}
    """
    schema = {
        "type": "object",
        "properties": raw_schema.get("properties") or {},
    }
    if "required" in raw_schema:
        schema["required"] = raw_schema["required"]
    return schema


def normalize_tool_result(result) -> object:
    """Normalise whatever the MCP SDK returns into a plain Python object."""
    content = result.content

    # Single content object with a .text attribute
    if hasattr(content, "text"):
        try:
            return json.loads(content.text)
        except (TypeError, json.JSONDecodeError):
            return content.text

    # List of content items (most common FastMCP response)
    if isinstance(content, list):
        normalized = []
        for item in content:
            if hasattr(item, "text"):
                try:
                    normalized.append(json.loads(item.text))
                except (TypeError, json.JSONDecodeError):
                    normalized.append(item.text)
            elif hasattr(item, "data"):
                normalized.append(item.data)
            else:
                normalized.append(str(item))
        return normalized[0] if len(normalized) == 1 else normalized

    # Fallback: try JSON decode of raw content
    try:
        return json.loads(content)
    except (TypeError, json.JSONDecodeError):
        return content


async def main():
    server_params_list = [
        StdioServerParameters(command=sys.executable, args=["weather_mcp.py"]),
        StdioServerParameters(command=sys.executable, args=["pollution_mcp.py"]),
        StdioServerParameters(command=sys.executable, args=["model_mcp.py"]),
    ]

    async with ClientSessionGroup() as session_group:
        for server_params in server_params_list:
            await session_group.connect_to_server(server_params)

        tools = session_group.tools

        # Build OpenAI-compatible tool definitions with sanitized schemas
        openai_tools = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": get_tool_description(tool.name, tool.description or ""),
                    "parameters": sanitize_input_schema(tool.inputSchema or {}),
                },
            }
            for tool in tools.values()
        ]

        user_prompt = input("Enter your prompt: ")
        messages = [
            {"role": "system", "content": build_system_prompt()},
            {"role": "user", "content": user_prompt},
        ]

        # Use "required" when the prompt is clearly tool-relevant so small local
        # models (qwen2.5:7b) can't skip tools with a plain text reply.
        tool_choice = "required" if is_tool_relevant(user_prompt) else "auto"
        retry_attempted = False

        while True:
            response = client.chat.completions.create(
                model=DEFAULT_MODEL,
                messages=messages,
                tools=openai_tools,
                tool_choice=tool_choice,
            )

            msg = response.choices[0].message

            # ── No tool calls issued ─────────────────────────────────────────
            if not msg.tool_calls:
                # First attempt with tool_choice="required" returned no calls —
                # this is unusual. Try once more with an explicit user nudge.
                if is_tool_relevant(user_prompt) and not retry_attempted:
                    print("\n[client] Model skipped tools on a tool-relevant prompt. Retrying with explicit nudge.")
                    messages.append({"role": "assistant", "content": msg.content or ""})
                    messages.append({
                        "role": "user",
                        "content": (
                            "You have access to live tools. Please call get_weather_bath, "
                            "get_pollution_bath, and/or predict_bath_pm25_forecast now "
                            "to answer my question — do not answer from memory."
                        ),
                    })
                    tool_choice = "required"
                    retry_attempted = True
                    continue

                # Final answer — no tools needed or retry exhausted
                print("\nFinal Answer:\n", msg.content)
                break

            # ── Tool calls issued ────────────────────────────────────────────
            print(f"\n[client] Model issued {len(msg.tool_calls)} tool call(s).")

            # Append the assistant message ONCE, containing all tool_calls
            messages.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.function.name,
                            "arguments": call.function.arguments,
                        },
                    }
                    for call in msg.tool_calls
                ],
            })

            # Execute every tool call and append one tool-result message each
            for call in msg.tool_calls:
                tool_name = call.function.name
                try:
                    args = json.loads(call.function.arguments)
                except json.JSONDecodeError:
                    args = {}

                print(f"  → Calling tool: {tool_name}  args={json.dumps(args)}")
                result = await session_group.call_tool(tool_name, args)
                normalized = normalize_tool_result(result)
                print(f"  ← Result: {json.dumps(normalized, indent=2, ensure_ascii=False)[:400]}…")

                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": json.dumps(normalized, ensure_ascii=False),
                })

            # After all tool results are in, let the model produce its final answer.
            # Switch back to "auto" so it can reply naturally without forcing more calls.
            tool_choice = "auto"


asyncio.run(main())
