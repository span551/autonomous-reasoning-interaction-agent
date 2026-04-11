"""
Intent Classification module
Supports:
  - ollama  : local models via Ollama (llama3.2, mistral, gemma2, etc.)
  - groq    : fast inference via Groq API
  - openai  : OpenAI API
"""
import os
import json
import re


INTENT_SYSTEM_PROMPT = """You are an intent classification engine for a voice-controlled AI agent.

Analyze the user's command and return a JSON object with this exact structure:
{
  "primary_intent": "<intent>",
  "compound_intents": ["<additional intents if compound command>"],
  "entities": {
    "filename": "<filename if mentioned, else inferred>",
    "language": "<programming language if code-related>",
    "content": "<content/topic to work with>",
    "description": "<brief description of what to do>"
  },
  "confidence": <0.0 to 1.0>
}

Supported intents:
- "create_file"   : create a new file (non-code, e.g. text, markdown)
- "write_code"    : generate code and save to a file
- "summarize"     : summarize provided text or describe summarization task
- "general_chat"  : answer a question, have a conversation, provide information
- "create_folder" : create a directory/folder

For compound commands (e.g. "summarize this and save to summary.txt"), set primary_intent to the
first action and list additional intents in compound_intents.

Rules:
- Always return valid JSON only. No preamble, no markdown backticks.
- For filenames: if not mentioned, infer a sensible one (e.g. "bubble_sort.py", "notes.txt")
- For write_code: detect the programming language from context
- confidence should reflect how certain you are about the classification

Examples:
User: "Create a Python file with a retry decorator"
{"primary_intent":"write_code","compound_intents":[],"entities":{"filename":"retry_decorator.py","language":"Python","content":"retry decorator","description":"Python retry decorator function"},"confidence":0.97}

User: "Make a folder called experiments"
{"primary_intent":"create_folder","compound_intents":[],"entities":{"filename":"experiments","language":null,"content":"","description":"Create folder named experiments"},"confidence":0.99}

User: "Summarize the following text and save it to a file: The quick brown fox..."
{"primary_intent":"summarize","compound_intents":["create_file"],"entities":{"filename":"summary.txt","language":null,"content":"The quick brown fox...","description":"Summarize text and save to file"},"confidence":0.95}
"""


def classify_intent(text: str, provider: str = "ollama", model: str = "llama3.2") -> dict:
    """
    Classify the intent of a transcribed voice command.
    
    Args:
        text: The transcribed text to classify
        provider: One of 'ollama', 'groq', 'openai'
        model: Model name to use
    
    Returns:
        Dict with keys: primary_intent, compound_intents, entities, confidence
    """
    if provider == "ollama":
        raw = _classify_ollama(text, model)
    elif provider == "groq":
        raw = _classify_groq(text, model)
    elif provider == "openai":
        raw = _classify_openai(text, model)
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")
    
    return _parse_intent_response(raw, text)


def _extract_json(text: str) -> str:
    """Extract JSON from text that might contain extra content."""
    # Try to find JSON object in the text
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        return match.group(0)
    return text


def _parse_intent_response(raw: str, original_text: str) -> dict:
    """Parse and validate the intent JSON response."""
    try:
        clean = _extract_json(raw.strip())
        data = json.loads(clean)
        
        # Ensure required fields
        data.setdefault("primary_intent", "general_chat")
        data.setdefault("compound_intents", [])
        data.setdefault("entities", {})
        data.setdefault("confidence", 0.5)
        
        # Store original text for tool use
        data["original_text"] = original_text
        
        return data
    except (json.JSONDecodeError, Exception) as e:
        # Graceful degradation: fall back to general_chat
        return {
            "primary_intent": "general_chat",
            "compound_intents": [],
            "entities": {"content": original_text, "description": "general conversation"},
            "confidence": 0.3,
            "parse_error": str(e),
            "original_text": original_text,
        }


def _classify_ollama(text: str, model: str) -> str:
    """Classify intent using Ollama local model."""
    try:
        import ollama
    except ImportError:
        raise ImportError(
            "ollama package not installed. Run: pip install ollama\n"
            "Also install Ollama from https://ollama.ai and pull a model:\n"
            "  ollama pull llama3.2"
        )
    
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": INTENT_SYSTEM_PROMPT},
            {"role": "user", "content": text}
        ],
        options={"temperature": 0.1}
    )
    return response["message"]["content"]


def _classify_groq(text: str, model: str) -> str:
    """Classify intent using Groq API."""
    try:
        from groq import Groq
    except ImportError:
        raise ImportError("groq package not installed. Run: pip install groq")
    
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable not set.")
    
    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=model or "llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": INTENT_SYSTEM_PROMPT},
            {"role": "user", "content": text}
        ],
        temperature=0.1,
        max_tokens=400,
    )
    return response.choices[0].message.content


def _classify_openai(text: str, model: str) -> str:
    """Classify intent using OpenAI API."""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("openai package not installed. Run: pip install openai")
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set.")
    
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model or "gpt-4o-mini",
        messages=[
            {"role": "system", "content": INTENT_SYSTEM_PROMPT},
            {"role": "user", "content": text}
        ],
        temperature=0.1,
        max_tokens=400,
    )
    return response.choices[0].message.content
