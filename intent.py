"""
Intent Classification via Groq LLM.
Returns a structured JSON object describing the user's intent.
"""
import os
import json
import re


INTENT_SYSTEM_PROMPT = """You are an intent classification engine for a voice-controlled AI agent.

Analyze the user's command and return a JSON object with EXACTLY this structure:
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
- Infer a sensible filename if one isn't mentioned (e.g. "bubble_sort.py", "notes.txt")
- Detect the programming language from context for write_code intents
- confidence reflects how certain you are about the classification

Examples:
User: "Create a Python file with a retry decorator"
{"primary_intent":"write_code","compound_intents":[],"entities":{"filename":"retry_decorator.py","language":"Python","content":"retry decorator","description":"Python retry decorator function"},"confidence":0.97}

User: "Make a folder called experiments"
{"primary_intent":"create_folder","compound_intents":[],"entities":{"filename":"experiments","language":null,"content":"","description":"Create folder named experiments"},"confidence":0.99}

User: "Summarize the following and save it to a file: The quick brown fox..."
{"primary_intent":"summarize","compound_intents":["create_file"],"entities":{"filename":"summary.txt","language":null,"content":"The quick brown fox...","description":"Summarize text and save to file"},"confidence":0.95}
"""


def classify_intent(text: str, model: str = "llama-3.1-8b-instant") -> dict:
    """
    Classify the intent of a transcribed voice command using Groq.

    Args:
        text:  The transcribed text to classify
        model: Groq model to use (default: llama-3.1-8b-instant)

    Returns:
        Dict with keys: primary_intent, compound_intents, entities, confidence, original_text
    """
    try:
        from groq import Groq
    except ImportError:
        raise ImportError("groq package not installed. Run: pip install groq")

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable is not set.")

    client = Groq(api_key=api_key)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": INTENT_SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        temperature=0.1,
        max_tokens=400,
    )

    raw = response.choices[0].message.content
    return _parse_response(raw, text)


def _extract_json(text: str) -> str:
    """Pull the first JSON object out of a string."""
    match = re.search(r'\{.*\}', text, re.DOTALL)
    return match.group(0) if match else text


def _parse_response(raw: str, original_text: str) -> dict:
    """Parse and validate the LLM's JSON response with graceful fallback."""
    try:
        data = json.loads(_extract_json(raw.strip()))
        data.setdefault("primary_intent", "general_chat")
        data.setdefault("compound_intents", [])
        data.setdefault("entities", {})
        data.setdefault("confidence", 0.5)
        data["original_text"] = original_text
        return data
    except Exception as e:
        # Graceful degradation: fall back to general_chat
        return {
            "primary_intent": "general_chat",
            "compound_intents": [],
            "entities": {"content": original_text, "description": "general conversation"},
            "confidence": 0.3,
            "parse_error": str(e),
            "original_text": original_text,
        }
