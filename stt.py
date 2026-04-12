"""
Speech-to-Text via Groq Whisper API (whisper-large-v3)
"""
import os


def transcribe_audio(audio_path: str) -> str:
    """
    Transcribe an audio file using Groq's Whisper API.

    Args:
        audio_path: Path to the audio file (.wav, .mp3, .m4a, etc.)

    Returns:
        Transcribed text string

    Raises:
        ImportError: if groq package isn't installed
        ValueError: if GROQ_API_KEY isn't set
    """
    try:
        from groq import Groq
    except ImportError:
        raise ImportError("groq package not installed. Run: pip install groq")

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY environment variable is not set.\n"
            "Get a free key at https://console.groq.com"
        )

    client = Groq(api_key=api_key)

    with open(audio_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model="whisper-large-v3",
            file=audio_file,
            response_format="text",
        )

    # Groq returns a plain string when response_format="text"
    result = transcription if isinstance(transcription, str) else transcription.text
    return result.strip()
