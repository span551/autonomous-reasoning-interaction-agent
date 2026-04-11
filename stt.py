"""
Speech-to-Text module
Supports:
  - groq        : Whisper via Groq API (fast, free tier available)
  - openai      : Whisper via OpenAI API
  - whisper_local: runs openai-whisper locally (requires CUDA or CPU)
"""
import os


def transcribe_audio(audio_path: str, provider: str = "groq") -> str:
    """
    Transcribe an audio file to text.
    
    Args:
        audio_path: Path to the audio file (.wav, .mp3, etc.)
        provider: One of 'groq', 'openai', 'whisper_local'
    
    Returns:
        Transcribed text string
    """
    if provider == "groq":
        return _transcribe_groq(audio_path)
    elif provider == "openai":
        return _transcribe_openai(audio_path)
    elif provider == "whisper_local":
        return _transcribe_whisper_local(audio_path)
    else:
        raise ValueError(f"Unknown STT provider: {provider}")


def _transcribe_groq(audio_path: str) -> str:
    """Transcribe using Groq's Whisper API."""
    try:
        from groq import Groq
    except ImportError:
        raise ImportError("groq package not installed. Run: pip install groq")
    
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable not set.")
    
    client = Groq(api_key=api_key)
    
    with open(audio_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model="whisper-large-v3",
            file=audio_file,
            response_format="text"
        )
    
    return transcription.strip() if isinstance(transcription, str) else transcription.text.strip()


def _transcribe_openai(audio_path: str) -> str:
    """Transcribe using OpenAI's Whisper API."""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("openai package not installed. Run: pip install openai")
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set.")
    
    client = OpenAI(api_key=api_key)
    
    with open(audio_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
        )
    
    return transcription.text.strip()


def _transcribe_whisper_local(audio_path: str) -> str:
    """
    Transcribe using local Whisper model.
    Requires: pip install openai-whisper
    First run downloads model weights (~244MB for 'base').
    """
    try:
        import whisper
    except ImportError:
        raise ImportError(
            "whisper package not installed. Run: pip install openai-whisper\n"
            "Also requires ffmpeg: apt-get install ffmpeg (Linux) or brew install ffmpeg (Mac)"
        )
    
    # Load model (cached after first use)
    # Options: tiny, base, small, medium, large
    # Use 'base' as default for balance of speed/accuracy
    model_size = os.getenv("WHISPER_MODEL", "base")
    model = whisper.load_model(model_size)
    
    result = model.transcribe(audio_path)
    return result["text"].strip()
