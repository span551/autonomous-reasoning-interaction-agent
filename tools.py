"""
Tool Execution module.
Handles: create_file, write_code, summarize, general_chat, create_folder.
Supports compound commands and chat context memory.

Safety: ALL file operations are restricted to the output/ directory.
"""
import os
import re
from pathlib import Path
from datetime import datetime

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)


# ── Safety ─────────────────────────────────────────────────────────────────────

def safe_filename(name: str) -> str:
    if not name:
        name = f"file_{datetime.now().strftime('%H%M%S')}.txt"
    name = Path(name).name
    name = re.sub(r'[^\w\-.]', '_', name)
    return name

def safe_path(filename: str) -> Path:
    return OUTPUT_DIR / safe_filename(filename)


# ── Main dispatcher ────────────────────────────────────────────────────────────

def execute_tool(intent_data: dict, confirmed: bool = False, chat_context: list = None) -> dict:
    """
    Execute the appropriate tool based on intent_data.

    Args:
        intent_data:  Dict returned by classify_intent()
        confirmed:    True after human-in-loop confirmation
        chat_context: Prior conversation turns for context-aware generation

    Returns:
        Dict with: status, message, output, action_taken, files_created
    """
    ctx = chat_context or []
    primary  = intent_data.get("primary_intent", "general_chat")
    compound = intent_data.get("compound_intents", [])

    result = _dispatch(primary, intent_data, ctx=ctx)

    for extra in compound:
        if extra != primary:
            extra_result = _dispatch(extra, intent_data, context=result, ctx=ctx)
            result["output"]        = (result.get("output", "") + "\n\n--- " +
                                       extra.upper() + " ---\n" + extra_result.get("output", ""))
            result["files_created"] = result.get("files_created", []) + extra_result.get("files_created", [])
            result["action_taken"]  = result.get("action_taken", "") + " + " + extra_result.get("action_taken", "")

    return result


def _dispatch(intent: str, intent_data: dict, context: dict = None, ctx: list = None) -> dict:
    handlers = {
        "create_file":   _handle_create_file,
        "write_code":    _handle_write_code,
        "summarize":     _handle_summarize,
        "general_chat":  _handle_general_chat,
        "create_folder": _handle_create_folder,
    }
    return handlers.get(intent, _handle_general_chat)(intent_data, context=context, ctx=ctx or [])


# ── Handlers ───────────────────────────────────────────────────────────────────

def _handle_create_file(intent_data: dict, context: dict = None, ctx: list = None) -> dict:
    entities    = intent_data.get("entities", {})
    filename    = safe_filename(entities.get("filename") or "document.txt")
    description = entities.get("description", "")
    original    = intent_data.get("original_text", description)

    content = _llm(
        system="You are a helpful writing assistant. Generate the file content requested. "
               "Return ONLY the content — no explanations, no preamble.",
        user=f"Generate content for a file based on this request:\n{original}\nDescription: {description}",
        ctx=ctx or []
    )

    is_text = filename.endswith((".md", ".txt"))
    header  = (f"# Created by ARIA\n# {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
               f"# Request: {original}\n\n") if is_text else ""

    safe_path(filename).write_text(header + content, encoding="utf-8")

    return {
        "status": "success",
        "message": f"File created: output/{filename}",
        "output": content,
        "action_taken": f"Created file → output/{filename}",
        "files_created": [filename],
    }


def _handle_write_code(intent_data: dict, context: dict = None, ctx: list = None) -> dict:
    entities = intent_data.get("entities", {})
    language = entities.get("language") or "Python"
    filename = safe_filename(entities.get("filename") or f"code_{datetime.now().strftime('%H%M%S')}.py")
    original = intent_data.get("original_text", entities.get("description", ""))

    ext_map = {
        "python":".py","javascript":".js","typescript":".ts","java":".java",
        "c++":".cpp","c":".c","go":".go","rust":".rs","bash":".sh",
        "html":".html","css":".css","sql":".sql","ruby":".rb","php":".php",
    }
    inferred_ext = ext_map.get(language.lower(), ".py")
    if not any(filename.endswith(e) for e in ext_map.values()):
        filename = Path(filename).stem + inferred_ext

    code = _llm(
        system=f"You are an expert {language} developer. Write clean, well-commented, "
               "production-quality code with docstrings and type hints. "
               "Return ONLY the code — no markdown fences, no explanations.",
        user=original,
        ctx=ctx or []
    )

    safe_path(filename).write_text(code, encoding="utf-8")

    return {
        "status": "success",
        "message": f"Code saved: output/{filename}",
        "output": code,
        "action_taken": f"Generated {language} code → output/{filename}",
        "files_created": [filename],
    }


def _handle_summarize(intent_data: dict, context: dict = None, ctx: list = None) -> dict:
    entities = intent_data.get("entities", {})
    content  = entities.get("content", "")
    original = intent_data.get("original_text", "")
    filename = safe_filename(entities.get("filename") or "summary.txt")

    if not content or len(content) < 20:
        content = original

    summary = _llm(
        system="You are a concise summarization expert. Provide a clear, well-structured summary.",
        user=f"Summarize the following:\n\n{content}",
        ctx=ctx or []
    )

    files_created = []
    compound = intent_data.get("compound_intents", [])
    if "create_file" in compound or "save" in original.lower():
        safe_path(filename).write_text(
            f"Summary by ARIA\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n{summary}",
            encoding="utf-8"
        )
        files_created.append(filename)

    return {
        "status": "success",
        "message": "Summarized successfully." + (f" Saved to output/{filename}" if files_created else ""),
        "output": summary,
        "action_taken": "Summarized text" + (f" → output/{filename}" if files_created else ""),
        "files_created": files_created,
    }


def _handle_general_chat(intent_data: dict, context: dict = None, ctx: list = None) -> dict:
    original = intent_data.get("original_text", "")

    response = _llm(
        system="You are ARIA, a helpful and knowledgeable AI voice assistant. "
               "Answer questions clearly and concisely. Be friendly and direct. "
               "You have memory of prior conversation turns — use them to give contextual answers.",
        user=original,
        ctx=ctx or []
    )

    return {
        "status": "success",
        "message": "Response generated.",
        "output": response,
        "action_taken": "General chat / Q&A",
        "files_created": [],
    }


def _handle_create_folder(intent_data: dict, context: dict = None, ctx: list = None) -> dict:
    entities    = intent_data.get("entities", {})
    folder_name = safe_filename(entities.get("filename") or f"folder_{datetime.now().strftime('%H%M%S')}")

    folder_path = OUTPUT_DIR / folder_name
    folder_path.mkdir(exist_ok=True)
    (folder_path / ".gitkeep").touch()

    return {
        "status": "success",
        "message": f"Folder created: output/{folder_name}/",
        "output": f"✓ Directory created: output/{folder_name}/",
        "action_taken": f"Created folder → output/{folder_name}/",
        "files_created": [],
    }


# ── Groq LLM helper ────────────────────────────────────────────────────────────

def _llm(system: str, user: str, ctx: list = None) -> str:
    """
    Call Groq with optional chat context for memory.
    ctx: list of prior {role, content} messages injected between system and user.
    """
    try:
        from groq import Groq
    except ImportError:
        return "[Error: groq package not installed. Run: pip install groq]"

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return "[Error: GROQ_API_KEY not set]"

    model = os.getenv("ARIA_LLM_MODEL", "llama-3.1-8b-instant")

    # Build message list: system → context history → current user request
    messages = [{"role": "system", "content": system}]
    if ctx:
        messages.extend(ctx[-20:])   # last 10 turns
    messages.append({"role": "user", "content": user})

    try:
        client   = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.7,
            max_tokens=2048,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[Groq API Error: {e}]"
