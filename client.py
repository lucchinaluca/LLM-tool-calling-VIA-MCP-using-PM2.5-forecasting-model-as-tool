import asyncio
import json
from mcp import ClientSessionGroup, StdioServerParameters

from openai import OpenAI

# Use Ollama (runs locally, free after initial download)
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="not-needed"
)

def get_tool_description(tool_name: str, default: str) -> str:
    if tool_name == "get_weather_bath":
        return (
            "Fetch Bath weather using Open-Meteo and return a short 12-hour hourly forecast summary. "
            "Includes temperature, wind, precipitation, and humidity fields."
        )
    if tool_name == "get_pollution_bath":
        return (
            "Fetch latest Bath pollution measurements from OpenAQ and return values keyed by pollutant name. "
            "Useful for deciding whether outdoor exercise is healthy."
        )
    return default

async def main():
    server_params_list = [
        StdioServerParameters(
            command="python",
            args=["weather_mcp.py"]
        ),
        StdioServerParameters(
            command="python",
            args=["pollution_mcp.py"]
        )
    ]

    async with ClientSessionGroup(server_params_list) as session_group:
        # Get tools
        tools = session_group.tools

        # Convert MCP tools to OpenAI format
        openai_tools = []
        for tool in tools:
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": get_tool_description(tool.name, tool.description),
                    "parameters": tool.inputSchema
                }
            })

        user_prompt = "Should I go for a run in Bath today? Consider weather and pollution."

        messages = [{"role": "user", "content": user_prompt}]

        while True:
            response = client.chat.completions.create(
                model="llama3",
                messages=messages,
                tools=openai_tools,
                tool_choice="auto"
            )

            msg = response.choices[0].message

            # If no tool call → final answer
            if not msg.tool_calls:
                print("\nFinal Answer:\n", msg.content)
                break

            # Handle tool calls
            for call in msg.tool_calls:
                tool_name = call.function.name
                args = json.loads(call.function.arguments)
                print(f"Tool requested: {tool_name}")
                print(f"Arguments: {json.dumps(args)}")

                result = await session_group.call_tool(tool_name, args)
                print("Tool result:", json.dumps(result.content, indent=2))

                messages.append(msg)
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": json.dumps(result.content)
                })

asyncio.run(main())
