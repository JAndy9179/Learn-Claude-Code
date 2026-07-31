import os
import ast, json, os, subprocess
from anthropic import Anthropic
from dotenv import load_dotenv
from pathlib import Path
from dataclasses import dataclass
from typing import Any


"""
Difference between OpenAI and Anthropic API:

1. OpenAI: 每一条 message 只有一个 content, 即使要调用工具，也是单独拆成 tool_calls 字段, content 和 tool_calls 平级共存：

{
  "role": "assistant",
  "content": "让我来查一下",
  "tool_calls": [
    {"id": "call_1", "type": "function", "function": {"name": "bash", "arguments": "..."}},
    {"id": "call_2", "type": "function", "function": {"name": "bash", "arguments": "..."}}
  ]
}

2. Anthropic: 一条 message 的 content 是一个 block 列表, 文本和工具调用可以共存，还能有多个工具调用。

{
  "role": "assistant",
  "content": [
    {"type": "text", "text": "让我查一下"},
    {"type": "tool_use", "id": "toolu_1", "name": "bash", "input": {"command": "ls"}},
    {"type": "tool_use", "id": "toolu_2", "name": "bash", "input": {"command": "pwd"}}
  ]
}


{
  "role": "user",
  "content": [
    {"type": "tool_result", "tool_use_id": "toolu_1", "content": "output1"},
    {"type": "tool_result", "tool_use_id": "toolu_2", "content": "output2"}
  ]
}

因此在 agent_loop 中可以给 results 中 append 多条 block, 然后再将 block 赋值给 content

"""

"""
Q & A

Q: todo_write 的调用？
A: SYSTEM 中提示模型需要在做子任务之前需要 plan 一下, 然后 update. 但随着对话变长, 模型的注意力会被最近的上下文稀释, 
   越往后模型越"看不到"这条指令. 因此手动在 loop 里面设置了 rounds_since_todo, 超过若干轮不用 todo_write 就再注入提示

Q: todo_list 会随着模型的一步一步执行而变化吗？
A: 是的, 模型在执行的过程中比如发i西安需要新增一步验证, 也会动态地向 todo_list 中添加新的任务

"""


# ── Settings ────────────────────────────


WORKDIR = Path.cwd()

load_dotenv(override=True)

client = Anthropic(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("ANTHROPIC_BASE_URL"),
)
MODEL = os.getenv("MODEL_ID")
SYSTEM = (
    f"You are a coding agent at {os.getcwd()}, your workspace is {WORKDIR}. "
    "Before starting any multi-step task, use todo_write to plan your steps. "
    "Update status as you go."
)
CURRENT_TODOS: list[dict] = []

# ── Tools definitions ────────────────────────────


def safe_path(p: str) -> Path:
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path


def run_read(path: str, limit: int | None = None) -> str:
    try:
        lines = safe_path(path).read_text().splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more lines)"]
        return "\n".join(lines)
    except Exception as e:
        return f"Error: {e}"


def run_write(path: str, content: str) -> str:
    try:
        file_path = safe_path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)
        return f"Wrote {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error: {e}"


def run_edit(path: str, old_text: str, new_text: str) -> str:
    try:
        file_path = safe_path(path)
        text = file_path.read_text()
        if old_text not in text:
            return f"Error: text not found in {path}"
        file_path.write_text(text.replace(old_text, new_text, 1))
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"


def run_glob(pattern: str) -> str:
    # 文件搜索
    import glob as g
    try:
        results = []
        for match in g.glob(pattern, root_dir=WORKDIR):
            if (WORKDIR / match).resolve().is_relative_to(WORKDIR):
                results.append(match)
        return "\n".join(results) if results else "(no matches)"
    except Exception as e:
        return f"Error: {e}"


def run_bash(command: str) -> str:
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        r = subprocess.run(command, shell=True, cwd=WORKDIR,
                           capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=120)
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"
    except (FileNotFoundError, OSError) as e:
        return f"Error: {e}"


def _normalize_todos(todos):
    if isinstance(todos, str):
        try:
            todos = json.loads(todos)
        except json.JSONDecodeError:
            try:
                todos = ast.literal_eval(todos)
            except (SyntaxError, ValueError):
                return None, "Error: todos must be a list or JSON array string"
    if not isinstance(todos, list):
        return None, "Error: todos must be a list"
    for i, t in enumerate(todos):
        if not isinstance(t, dict):
            return None, f"Error: todos[{i}] must be an object"
        if "content" not in t or "status" not in t:
            return None, f"Error: todos[{i}] missing 'content' or 'status'"
        if t["status"] not in ("pending", "in_progress", "completed"):
            return None, f"Error: todos[{i}] has invalid status '{t['status']}'"
    return todos, None


def run_todo_write(todos: list) -> str:
    # 修改的是全局变量 CURRENT_TODOS
    global CURRENT_TODOS
    todos, error = _normalize_todos(todos)
    if error:
        return error
    CURRENT_TODOS = todos

    # 展示 todo 列表
    lines = ["\n\033[33m## Current Tasks\033[0m"]
    for t in CURRENT_TODOS:
        icon = {"pending": " ", "in_progress": "\033[36m▸\033[0m", "completed": "\033[32m✓\033[0m"}[t["status"]]
        lines.append(f"  [{icon}] {t['content']}")
    print("\n".join(lines))
    return f"Updated {len(CURRENT_TODOS)} tasks"


TOOLS = [
    {"name": "bash", "description": "Run a shell command.",
     "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
    {"name": "read_file", "description": "Read file contents.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["path"]}},
    {"name": "write_file", "description": "Write content to a file.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}},
    {"name": "edit_file", "description": "Replace exact text in a file once.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["path", "old_text", "new_text"]}},
    {"name": "glob", "description": "Find files matching a glob pattern.",
     "input_schema": {"type": "object", "properties": {"pattern": {"type": "string"}}, "required": ["pattern"]}},
     # todo write tools: 这里面定义了, 输入 todo_write 的参数格式必须是 [{'content': , 'status': }], 且 status 必须是 'pending', 'in_progress' 或 'completed' 中的一个
    {
        "name": "todo_write", 
        "description": "Create and manage a task list for your current coding session.",
        "input_schema": {
            "type": "object", 
            "properties": {
                "todos": {
                    "type": "array", 
                    "items": {
                        "type": "object", 
                        "properties": {
                            "content": {"type": "string"}, 
                            "status": {
                                "type": "string", 
                                "enum": ["pending", "in_progress", "completed"]
                            }
                        }, 
                        "required": ["content", "status"]
                    }
                }}
            , 
            "required": ["todos"]
        }
    },
]

TOOL_HANDLERS = {
    "bash":       lambda **kw: run_bash(kw["command"]),
    "read_file":  lambda **kw: run_read(kw["path"], kw.get("limit")),
    "write_file": lambda **kw: run_write(kw["path"], kw["content"]),
    "edit_file":  lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"]),
    "glob":       lambda **kw: run_glob(kw["pattern"]),
    "todo_write": lambda **kw: run_todo_write(kw["todos"]),
}


# ── Unified Context Object ─────────────────────────


@dataclass
class HookContent:
    """统一传给所有 hook 的上下文对象, 用于统一不同钩子函数之间传入参数"""
    hook_event_name: str                        # 事件名
    tool_name: str | None = None                # 工具名
    tool_input: dict[str, Any] | None = None    # 工具输入
    tool_output: str | None = None              # 工具输出
    user_query: str | None = None               # 用户输入
    message: list | None = None                 # 消息历史


@dataclass
class HookResult:
    """所有钩子函数统一的返回结果类型"""
    action: str = "continue"                    # "continue" | "block" | "force_continue"
    message: str | None = None                  # 附带消息


# —— Hook System ────────────────────────────────────


HOOKS = {
    "UserPromptSubmit": [],
    "PreToolUse": [],
    "PostToolUse": [],
    "Stop": [],
}


def register_hook(event: str, callback):
    HOOKS[event].append(callback)


def trigger_hooks(event: str, ctx: HookContent) -> HookResult:
    final = HookResult()
    for callback in HOOKS[event]:
        result = callback(ctx)
        if result is None:
            continue
        if result.action == "block":
            return result
        elif result.action == "force_continue":
            final = result
    return final


DENY_LIST = ["rm -rf /", "sudo", "shutdown", "reboot", "mkfs", "dd if="]
DESTRUCTIVE = ["rm ", "> /etc/", "chmod 777"]


def permission_hook(ctx: HookContent) -> HookResult:
    """PreToolUse: s03 check_permission() logic moved here."""
    if ctx.tool_name == "bash":
        for pattern in DENY_LIST:
            if pattern in ctx.tool_input.get("command", ""):
                print(f"\n\033[31m⛔ Blocked: '{pattern}'\033[0m")
                return HookResult(action="block", message="Permission denied by deny list")
        for kw in DESTRUCTIVE:
            if kw in ctx.tool_input.get("command", ""):
                print(f"\n\033[33m⚠  Potentially destructive command\033[0m")
                print(f"   Tool: {ctx.tool_name}({ctx.tool_input})")
                choice = input("   Allow? [y/N] ").strip().lower()
                if choice not in ("y", "yes"):
                    return HookResult(action="block", message="Permission denied by user")
    if ctx.tool_name in ("write_file", "edit_file"):
        path = ctx.tool_input.get("path", "")
        if not (WORKDIR / path).resolve().is_relative_to(WORKDIR):
            print(f"\n\033[33m⚠  Writing outside workspace\033[0m")
            print(f"   Tool: {ctx.tool_name}({ctx.tool_input})")
            choice = input("   Allow? [y/N] ").strip().lower()
            if choice not in ("y", "yes"):
                return HookResult(action="block", message="Permission denied by user")
    return None


def log_hook(ctx: HookContent) -> HookResult:
    """PreToolUse: log every tool call."""
    args_preview = str(list(ctx.tool_input.values())[:2])[:60]
    print(f"\033[90m[HOOK] {ctx.tool_name}({args_preview})\033[0m")
    return None


def large_output_hook(ctx: HookContent) -> HookResult:
    """PostToolUse: warn on large output."""
    if ctx.tool_output and len(str(ctx.tool_output)) > 100000:
        print(f"\033[33m[HOOK] ⚠ Large output from {ctx.tool_name}: {len(str(ctx.tool_output))} chars\033[0m")
    return None


def context_inject_hook(ctx: HookContent) -> HookResult:
    """UserPromptSubmit: inject context"""
    print(f"\033[90m[HOOK] UserPromptSubmit: working in {WORKDIR}\033[0m")
    return None


def summary_hook(ctx: HookContent) -> HookResult:
    """Stop: count tool calls"""
    tool_count = sum(
        1 
        for m in ctx.message
        for b in (m.get("content") if isinstance(m.get("content"), list) else [])
        if isinstance(b, dict) and b.get("type") == "tool_result"
    )
    print(f"\033[90m[HOOK] Stop: session used {tool_count} tool calls\033[0m")
    return None


register_hook("UserPromptSubmit", context_inject_hook)
register_hook("PreToolUse", permission_hook)
register_hook("PreToolUse", log_hook)
register_hook("PostToolUse", large_output_hook)
register_hook("Stop", summary_hook)


# ── Agent loop ────────────────────────────────────────


def agent_loop(messages: list):
    rounds_since_todo = 0
    while True:
        response = client.messages.create(
            model=MODEL, system=SYSTEM, messages=messages,
            tools=TOOLS, max_tokens=8000,
        )

        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            # Stop: 需要退出 loop 时（即触发 Stop 钩子，这里指的是工具调用结束时）需要执行的操作
            force = trigger_hooks("Stop", HookContent(
                hook_event_name="Stop", 
                message=messages
            ))
            if force.action == "force_continue":
                messages.append({"role": "user", "content": force.message or "continue"})
                continue
            return

        results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            
            # PreToolUse: 工具调用前需要执行的一些操作
            res = trigger_hooks("PreToolUse", HookContent(
                hook_event_name="PreToolUse", 
                tool_name=block.name, 
                tool_input=block.input
            ))
            if res.action == "block":
                results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": res.message or "Blocked by hook",
                    "is_error": True,
                })
                continue

            handler = TOOL_HANDLERS.get(block.name)
            output = handler(**block.input) if handler else f"Unknown: {block.name}"
            
            # PostToolUse: 工具调用后需要执行的一些操作
            trigger_hooks("PostToolUse", HookContent(
                hook_event_name="PostToolUse", 
                tool_name=block.name, 
                tool_output=output
            ))
            results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": output,
            })

            if block.name == "todo_write":
                rounds_since_todo = 0

        rounds_since_todo += 1
        if rounds_since_todo >= 3:
            results.append({
                "type": "text",
                "text": "<reminder>Update your todos.</reminder>",
            })
            rounds_since_todo = 0

        messages.append({"role": "user", "content": results})


if __name__ == "__main__":
    print("s05: TodoWrite — plan before execute")
    print("输入问题，回车发送。输入 q 退出。\n")

    history = []
    while True:
        try:
            query = input("\033[36ms04 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break

        if query.strip().lower() in ("q", "exit", ""):
            break
        
        # UserPromptSubmit: 用户输入 promt 后需要执行的一些操作
        trigger_hooks("UserPromptSubmit", HookContent(
            hook_event_name="UserPromptSubmit", 
            user_query=query
        ))

        history.append({"role": "user", "content": query})
        agent_loop(history)
        
        for block in history[-1]["content"]:
            if getattr(block, "type", None) == "text":
                print(block.text)
        print()