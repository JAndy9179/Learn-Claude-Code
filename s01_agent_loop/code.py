import os
import subprocess
from anthropic import Anthropic
from dotenv import load_dotenv


"""
anthropic response format:

Message(
    id='818d589b-f176-4836-a232-9df8e92d4cec', 
    container=None, 
    content=[
        ThinkingBlock(
            signature='818d589b-f176-4836-a232-9df8e92d4cec', 
            thinking='The file was created successfully. Let me verify it.', 
            type='thinking'
        ), 
        ToolUseBlock(
            id='call_00_7Bd9bRzSMpIbKOyi0TzX9237', 
            caller=None, 
            input={'command': 'python hello.py'}, 
            name='bash', 
            type='tool_use'
        )
    ], 
    model='deepseek-v4-pro', 
    role='assistant', 
    stop_details=None, 
    stop_reason='tool_use',
    stop_sequence=None, 
    type='message', 
    usage=Usage(cache_creation=None, cache_creation_input_tokens=0, cache_read_input_tokens=384, inference_geo=None, input_tokens=16, output_tokens=56, output_tokens_details=None, server_tool_use=None, service_tier='standard')
)
"""


# ── Settings ────────────────────────────


load_dotenv(override=True)

client = Anthropic(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("ANTHROPIC_BASE_URL"),
)
MODEL = os.getenv("MODEL_ID")

SYSTEM = f"You are a coding agent at {os.getcwd()}. Use bash to solve tasks. Act, don't explain."


# ── Tool definition: just bash ────────────────────────────


TOOLS = [
    {
        "name": "bash",
        "description": "Run a shell command.",
        "input_schema": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    }
]


# ── Tool execution ────────────────────────────────────────
def run_bash(command: str) -> str:
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        r = subprocess.run(command, shell=True, cwd=os.getcwd(),
                           capture_output=True, text=True, timeout=120)
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"
    except (FileNotFoundError, OSError) as e:
        return f"Error: {e}"


# ── The core pattern: a while loop that calls tools until the model stops ──
def agent_loop(messages: list):
    while True:
        response = client.messages.create(
            model=MODEL, system=SYSTEM, messages=messages,
            tools=TOOLS, max_tokens=8000,
        )

        messages.append({"role": "assistant", "content": response.content})
        print(response)

        if response.stop_reason != "tool_use":
            return

        results = []
        for block in response.content:
            if block.type == "tool_use":
                print(f"\033[33m$ {block.input['command']}\033[0m")
                output = run_bash(block.input["command"])
                print(output[:200])
                results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": output,
                })

        messages.append({"role": "user", "content": results})


if __name__ == "__main__":
    print("s01: Agent Loop")
    print("输入问题，回车发送。输入 q 退出。\n")

    history = []
    while True:
        try:
            query = input("\033[36ms01 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break

        if query.strip().lower() in ("q", "exit", ""):
            break

        history.append({"role": "user", "content": query})
        agent_loop(history)
        response_content = history[-1]["content"]
        if isinstance(response_content, list):
            for block in response_content:
                if getattr(block, "type", None) == "text":
                    print(block.text)
        print()