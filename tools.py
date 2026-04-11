"""
Tool Execution module
Handles: create_file, write_code, summarize, general_chat, create_folder
Supports compound commands (multiple intents in sequence).

Safety: ALL file operations are restricted to the output/ directory.
"""
import os
import re
from pathlib import Path
from datetime import datetime

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)


# ── Safety ────────────────────────────────────────────────────────────────────

def safe_filename(name: str) -> str:
    """Sanitize and ensure filename stays within output/."""
    if not name:
        name = f"file_{datetime.now().strftime('%H%M%S')}.txt"
    # Remove path traversal
    name = Path(name).name
    # Remove unsafe chars
    name = re.sub(r'[^\w\-.]', '_', name)
    return name


def safe_path(filename: str) -> Path:
    """Return a safe path inside output/."""
    return OUTPUT_DIR / safe_filename(filename)


# ── Main dispatcher ───────────────────────────────────────────────────────────

def execute_tool(intent_data: dict, confirmed: bool = False) -> dict:
    """
    Execute the appropriate tool based on intent_data.
    
    Args:
        intent_data: Dict from classify_intent()
        confirmed: For human-in-loop, whether user has confirmed file ops
    
    Returns:
        Dict with: status, message, output, action_taken, files_created
    """
    primary = intent_data.get("primary_intent", "general_chat")
    compound = intent_data.get("compound_intents", [])
    
    # Execute primary intent
    result = _dispatch(primary, intent_data)
    
    # Execute compound intents
    for extra_intent in compound:
        if extra_intent != primary:
            extra_result = _dispatch(extra_intent, intent_data, context=result)
            # Merge results
            result["output"] = result.get("output", "") + "\n\n--- " + extra_intent.upper() + " ---\n" + extra_result.get("output", "")
            result["files_created"] = result.get("files_created", []) + extra_result.get("files_created", [])
            result["action_taken"] = result.get("action_taken", "") + " + " + extra_result.get("action_taken", "")
    
    return result


def _dispatch(intent: str, intent_data: dict, context: dict = None) -> dict:
    """Route to the appropriate handler."""
    handlers = {
        "create_file": _handle_create_file,
        "write_code": _handle_write_code,
        "summarize": _handle_summarize,
        "general_chat": _handle_general_chat,
        "create_folder": _handle_create_folder,
    }
    handler = handlers.get(intent, _handle_general_chat)
    return handler(intent_data, context=context)


# ── Handlers ──────────────────────────────────────────────────────────────────

def _handle_create_file(intent_data: dict, context: dict = None) -> dict:
    """Create a new text/markdown file with generated content."""
    entities = intent_data.get("entities", {})
    filename = safe_filename(entities.get("filename") or "document.txt")
    description = entities.get("description", "")
    content_hint = entities.get("content", "")
    original = intent_data.get("original_text", description)
    
    # Generate content via LLM
    content = _generate_content(
        prompt=f"Generate the content for a file based on this request: {original}\n"
               f"File description: {description}\nContent topic: {content_hint}\n"
               f"Return only the file content, nothing else.",
        intent_data=intent_data
    )
    
    # Add header comment
    header = f"# Created by ARIA Voice Agent\n# {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n# Request: {original}\n\n"
    if not filename.endswith((".md", ".txt")):
        header = ""
    
    filepath = safe_path(filename)
    filepath.write_text(header + content, encoding="utf-8")
    
    return {
        "status": "success",
        "message": f"File created: output/{filename}",
        "output": content,
        "action_taken": f"Created file: output/{filename}",
        "files_created": [filename],
    }


def _handle_write_code(intent_data: dict, context: dict = None) -> dict:
    """Generate code and save it to a file."""
    entities = intent_data.get("entities", {})
    language = entities.get("language", "Python")
    filename = safe_filename(entities.get("filename") or f"code_{datetime.now().strftime('%H%M%S')}.py")
    description = entities.get("description", "")
    original = intent_data.get("original_text", description)
    
    # Ensure correct extension
    ext_map = {
        "python": ".py", "javascript": ".js", "typescript": ".ts",
        "java": ".java", "c++": ".cpp", "c": ".c", "go": ".go",
        "rust": ".rs", "bash": ".sh", "html": ".html", "css": ".css",
        "sql": ".sql", "ruby": ".rb", "php": ".php",
    }
    inferred_ext = ext_map.get(language.lower(), ".py")
    if not any(filename.endswith(e) for e in ext_map.values()):
        stem = Path(filename).stem
        filename = stem + inferred_ext
    
    code = _generate_code(
        request=original,
        language=language,
        intent_data=intent_data
    )
    
    filepath = safe_path(filename)
    filepath.write_text(code, encoding="utf-8")
    
    return {
        "status": "success",
        "message": f"Code saved: output/{filename}",
        "output": code,
        "action_taken": f"Generated {language} code → output/{filename}",
        "files_created": [filename],
    }


def _handle_summarize(intent_data: dict, context: dict = None) -> dict:
    """Summarize provided content."""
    entities = intent_data.get("entities", {})
    content = entities.get("content", "")
    original = intent_data.get("original_text", "")
    filename = safe_filename(entities.get("filename") or "summary.txt")
    
    # If no explicit content in entities, use original transcription minus the command
    if not content or len(content) < 20:
        content = original
    
    summary = _generate_summary(content, intent_data=intent_data)
    
    # Check if compound intent wants to save
    compound = intent_data.get("compound_intents", [])
    files_created = []
    if "create_file" in compound or "save" in original.lower():
        filepath = safe_path(filename)
        save_text = f"Summary generated by ARIA\nDate: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\nOriginal request: {original}\n\n{summary}"
        filepath.write_text(save_text, encoding="utf-8")
        files_created.append(filename)
    
    return {
        "status": "success",
        "message": "Text summarized successfully.",
        "output": summary,
        "action_taken": "Summarized text" + (f" → saved to output/{filename}" if files_created else ""),
        "files_created": files_created,
    }


def _handle_general_chat(intent_data: dict, context: dict = None) -> dict:
    """Handle general conversation/Q&A."""
    original = intent_data.get("original_text", "")
    
    response = _generate_chat_response(original, intent_data=intent_data)
    
    return {
        "status": "success",
        "message": "Response generated.",
        "output": response,
        "action_taken": "General chat / Q&A",
        "files_created": [],
    }


def _handle_create_folder(intent_data: dict, context: dict = None) -> dict:
    """Create a folder inside output/."""
    entities = intent_data.get("entities", {})
    folder_name = safe_filename(entities.get("filename") or f"folder_{datetime.now().strftime('%H%M%S')}")
    
    folder_path = OUTPUT_DIR / folder_name
    folder_path.mkdir(exist_ok=True)
    (folder_path / ".gitkeep").touch()
    
    return {
        "status": "success",
        "message": f"Folder created: output/{folder_name}/",
        "output": f"✓ Directory created: output/{folder_name}/",
        "action_taken": f"Created folder: output/{folder_name}/",
        "files_created": [],
    }


# ── LLM helpers ───────────────────────────────────────────────────────────────

def _call_llm(prompt: str, system: str, intent_data: dict) -> str:
    """Generic LLM call that respects the provider from intent classification."""
    # The intent_data might carry provider info if set during classification
    # We fall back to trying each provider in order
    
    # Try to detect which provider to use from environment
    provider = os.getenv("ARIA_LLM_PROVIDER", "ollama")
    model = os.getenv("ARIA_LLM_MODEL", "llama3.2")
    
    if provider == "ollama":
        return _call_ollama(prompt, system, model)
    elif provider == "groq":
        return _call_groq(prompt, system, model)
    elif provider == "openai":
        return _call_openai(prompt, system, model)
    else:
        return _call_ollama(prompt, system, model)


def _call_ollama(prompt: str, system: str, model: str) -> str:
    try:
        import ollama
        response = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt}
            ],
            options={"temperature": 0.7}
        )
        return response["message"]["content"]
    except Exception as e:
        return f"[LLM Error: {e}]\n\nFallback: Unable to generate content. Please check your Ollama setup."


def _call_groq(prompt: str, system: str, model: str) -> str:
    try:
        from groq import Groq
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        response = client.chat.completions.create(
            model=model or "llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=2000,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"[Groq Error: {e}]"


def _call_openai(prompt: str, system: str, model: str) -> str:
    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model=model or "gpt-4o-mini",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=2000,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"[OpenAI Error: {e}]"


def _generate_code(request: str, language: str, intent_data: dict) -> str:
    system = (
        f"You are an expert {language} developer. Generate clean, well-commented, production-quality code. "
        "Return ONLY the code with no explanations, no markdown backticks, no preamble. "
        "Include docstrings and type hints where appropriate."
    )
    return _call_llm(request, system, intent_data)


def _generate_content(prompt: str, intent_data: dict) -> str:
    system = (
        "You are a helpful assistant. Generate clear, well-structured content as requested. "
        "Return only the content itself, no meta-commentary."
    )
    return _call_llm(prompt, system, intent_data)


def _generate_summary(content: str, intent_data: dict) -> str:
    system = (
        "You are a concise summarization expert. Provide a clear, structured summary. "
        "Highlight key points. Be accurate and complete."
    )
    prompt = f"Please summarize the following:\n\n{content}"
    return _call_llm(prompt, system, intent_data)


def _generate_chat_response(text: str, intent_data: dict) -> str:
    system = (
        "You are ARIA, a helpful and knowledgeable AI voice assistant. "
        "Answer questions clearly and concisely. Be friendly and direct."
    )
    return _call_llm(text, system, intent_data)
