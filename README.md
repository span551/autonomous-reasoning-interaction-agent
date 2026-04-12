# 🎙️ ARIA — Autonomous Reasoning & Interaction Agent

A voice-controlled local AI agent powered entirely by **Groq API** — fast, free-tier Whisper for transcription and blazing LLM inference for intent + execution.


 project Link :https://autonomous-reasoning-interaction-agent.streamlit.app/
---

## 🏗️ Architecture

```
Audio Input (mic / file upload / text debug)
        │
        ▼
  ┌─────────────────────────────────────────┐
  │  STT: Groq Whisper (whisper-large-v3)   │
  └─────────────────────────────────────────┘
        │ transcribed text
        ▼
  ┌─────────────────────────────────────────┐
  │  Intent Classification: Groq LLM        │  ← llama-3.1-8b-instant (default)
  └─────────────────────────────────────────┘
        │ structured JSON intent
        ▼
  ┌─────────────────────────────────────────┐
  │  Tool Executor (local file ops, codegen) │
  └─────────────────────────────────────────┘
        │
        ▼
  ┌─────────────────────────────────────────┐
  │  Streamlit UI                            │
  └─────────────────────────────────────────┘
```

### Key files

| File | Purpose |
|------|---------|
| `app.py` | Streamlit UI — input tabs, pipeline trace, results, download buttons |
| `stt.py` | Speech-to-Text via Groq Whisper (`whisper-large-v3`) |
| `intent.py` | Intent classification via Groq LLM → structured JSON |
| `tools.py` | Tool executor for all intents; LLM generation also via Groq |
| `output/` | **All generated files go here — never outside this directory** |

---

## ⚙️ Setup

### 1. Clone & install

```bash
git clone https://github.com/span551/autonomous-reasoning-interaction-agent.git
cd autonomous-reasoning-interaction-agent
python -m pip install -r requirements.txt
```

### 2. Get your Groq API key (free)

1. Go to [console.groq.com](https://console.groq.com)
2. Sign up / log in
3. Create an API key

You can enter it directly in the sidebar at runtime — no `.env` file needed.
Or set it as an environment variable:

```bash
export GROQ_API_KEY=gsk_your_key_here
```

### 3. Run

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## 🎯 Supported Intents

| Intent | Example Command | Action |
|--------|----------------|--------|
| `write_code` | "Create a Python file with a bubble sort function" | Generates code → `output/bubble_sort.py` |
| `create_file` | "Write a README for a Flask project" | Generates content → `output/README.md` |
| `summarize` | "Summarize: The mitochondria is the powerhouse..." | Returns bullet-point summary |
| `create_folder` | "Make a folder called experiments" | Creates `output/experiments/` |
| `general_chat` | "What is a binary search tree?" | Answers conversationally |

### Compound commands
> "Summarize this text and save it to summary.txt"  
> "Write a Python web scraper and create a README for it"

---

## ✨ Features

- **Mic + file upload** — record live or upload .wav/.mp3/.m4a
- **Text debug mode** — bypass STT to test intent classification directly
- **Human-in-the-loop** — optional confirm prompt before any file operation
- **Session history** — all commands logged in the sidebar with timestamps
- **Compound commands** — multi-intent handling in one utterance
- **Graceful degradation** — falls back to `general_chat` on unknown intent
- **Model selector** — switch between Groq models in the sidebar
- **Download buttons** — instantly download any generated file
- **Pipeline timing** — latency shown per stage (STT / intent / exec)

---

## 🔧 Why Groq for everything?

Groq's LPU hardware delivers **extremely low latency** — Whisper transcription typically completes in 1–2 seconds, and LLM responses in under a second for 8B models. The free tier is generous enough for development and demos. This means no GPU required on your local machine.

---

## 📁 Output safety

All file writes are **restricted to `output/`**. Path traversal attempts are sanitized in `tools.py` before any file operation.

---

## 🚀 Bonus features implemented

- ✅ Compound commands
- ✅ Human-in-the-loop confirmation
- ✅ Graceful degradation on unknown intents
- ✅ Session memory / history sidebar
- ✅ Per-stage latency benchmarking in UI
