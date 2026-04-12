"""
Intent Classification via Groq LLM.
Accepts chat_context (list of prior {role, content} messages) for memory across turns.
"""
import os
import json
import re


INTENT_SYSTEM_PROMPT = """You are an intent classification engine for a voice-controlled AI agent.

You have access to the conversation history between the user and the assistant. Use it to resolve
references like "that", "it", "the previous file", "do the same but in JavaScript", etc.

Analyze the user's latest command and return a JSON object with EXACTLY this structure:
{
  "primary_intent": "<intent>",
  "compound_intents": ["<additional intents if compound command>"],
  "entities": {
    "filename": "<filename if mentioned, else infer a sensible one>",
    "language": "<programming language if code-related, else null>",
    "content": "<content or topic to work with>",
    "description": "<brief description of what to do>"
  },
  "confidence": <0.0 to 1.0>
}

Supported intents:
- "create_file"   : create a new non-code file (text, markdown, etc.)
- "write_code"    : generate code and save to a file
- "summarize"     : summarize provided text
- "general_chat"  : answer a question or have a conversation
- "create_folder" : create a directory/folder

For compound commands (e.g. "summarize this and save to summary.txt"), set primary_intent to the
first action and list additional intents in compound_intents.

Rules:
- Return ONLY valid JSON. No preamble, no markdown fences, no explanation.
- Use conversation history to resolve ambiguous references.
- Infer a sensible filename if one isn't mentioned (e.g. "bubble_sort.py", "notes.txt")
- confidence reflects how certain you are about the classification
"""


def classify_intent(
    text: str,
    model: str = "llama-3.1-8b-instant",
    chat_context: list = None
) -> dict:
    """
    Classify intent of a voice command using Groq, with optional conversation memory.

    Args:
        text:         Transcribed text to classify
        model:        Groq model name
        chat_context: List of prior {role, content} dicts for memory

    Returns:
        Dict with primary_intent, compound_intents, entities, confidence, original_text
    """
    try:
        from groq import Groq
    except ImportError:
        raise ImportError("groq package not installed. Run: pip install groq")

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable is not set.")

    client = Groq(api_key=api_key)

    # Build messages: system + prior context + current user message
    messages = [{"role": "system", "content": INTENT_SYSTEM_PROMPT}]

    # Inject up to last 10 turns of chat context (20 messages)
    if chat_context:
        messages.extend(chat_context[-20:])

    messages.append({"role": "user", "content": text})

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.1,
        max_tokens=400,
    )

    raw = response.choices[0].message.content
    return _parse_response(raw, text)


def _extract_json(text: str) -> str:
    match = re.search(r'\{.*\}', text, re.DOTALL)
    return match.group(0) if match else text


def _parse_response(raw: str, original_text: str) -> dict:
    try:
        data = json.loads(_extract_json(raw.strip()))
        data.setdefault("primary_intent", "general_chat")
        data.setdefault("compound_intents", [])
        data.setdefault("entities", {})
        data.setdefault("confidence", 0.5)
        data["original_text"] = original_text
        return data
    except Exception as e:
        return {
            "primary_intent": "general_chat",
            "compound_intents": [],
            "entities": {"content": original_text, "description": "general conversation"},
            "confidence": 0.3,
            "parse_error": str(e),
            "original_text": original_text,
        }
