import streamlit as st
import tempfile
import os
import json
from pathlib import Path
from datetime import datetime
import time

from stt import transcribe_audio
from intent import classify_intent
from tools import execute_tool

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ARIA – Voice AI Agent",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:ital,wght@0,400;0,700;1,400&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg: #0a0a0f;
    --surface: #111118;
    --surface2: #1a1a24;
    --accent: #7c6af7;
    --accent2: #c084fc;
    --success: #34d399;
    --warn: #fbbf24;
    --err: #f87171;
    --text: #e2e2f0;
    --muted: #6b6b8a;
    --border: #2a2a3a;
}

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: var(--bg); color: var(--text); }

/* Sidebar */
[data-testid="stSidebar"] {
    background: var(--surface);
    border-right: 1px solid var(--border);
}

/* Headers */
h1, h2, h3 { font-family: 'Space Mono', monospace; }

/* Cards */
.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px 24px;
    margin: 10px 0;
}
.card-accent { border-left: 3px solid var(--accent); }
.card-success { border-left: 3px solid var(--success); }
.card-warn { border-left: 3px solid var(--warn); }
.card-err { border-left: 3px solid var(--err); }

/* Tags */
.tag {
    display: inline-block;
    padding: 3px 12px;
    border-radius: 20px;
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
.tag-intent { background: #2d2060; color: var(--accent2); border: 1px solid #4a3a90; }
.tag-success { background: #0d2e22; color: var(--success); border: 1px solid #1a5a3a; }
.tag-warn { background: #2e2100; color: var(--warn); border: 1px solid #5a3e00; }

/* Pipeline steps */
.step {
    display: flex;
    align-items: flex-start;
    gap: 16px;
    padding: 14px 0;
    border-bottom: 1px solid var(--border);
}
.step:last-child { border-bottom: none; }
.step-num {
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    color: var(--accent);
    background: #1e1840;
    border: 1px solid #3d2e80;
    border-radius: 4px;
    padding: 2px 8px;
    white-space: nowrap;
    margin-top: 2px;
}
.step-content { flex: 1; }
.step-label { font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 4px; }
.step-value { font-family: 'Space Mono', monospace; font-size: 13px; color: var(--text); word-break: break-word; }

/* History item */
.history-item {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 12px 16px;
    margin: 6px 0;
    cursor: pointer;
    transition: border-color 0.2s;
}
.history-item:hover { border-color: var(--accent); }
.history-ts { font-family: 'Space Mono', monospace; font-size: 10px; color: var(--muted); }
.history-text { font-size: 13px; color: var(--text); margin: 4px 0; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }

/* Code block */
.code-out {
    background: #0d0d14;
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 16px;
    font-family: 'Space Mono', monospace;
    font-size: 12px;
    line-height: 1.6;
    color: #b4b4d4;
    overflow-x: auto;
    white-space: pre-wrap;
    word-break: break-word;
}

/* Streamlit button overrides */
.stButton > button {
    font-family: 'Space Mono', monospace !important;
    font-size: 12px !important;
    letter-spacing: 0.05em !important;
    background: var(--accent) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 8px 20px !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

/* Divider */
hr { border-color: var(--border); margin: 24px 0; }

/* Waveform animation */
@keyframes wave {
    0%, 100% { height: 4px; }
    50% { height: 20px; }
}
.waveform { display: flex; align-items: center; gap: 3px; height: 28px; }
.wave-bar {
    width: 3px;
    background: var(--accent);
    border-radius: 2px;
    animation: wave 1.2s ease-in-out infinite;
}
.wave-bar:nth-child(2) { animation-delay: 0.1s; }
.wave-bar:nth-child(3) { animation-delay: 0.2s; }
.wave-bar:nth-child(4) { animation-delay: 0.3s; }
.wave-bar:nth-child(5) { animation-delay: 0.4s; }
.wave-bar:nth-child(6) { animation-delay: 0.5s; }
.wave-bar:nth-child(7) { animation-delay: 0.3s; }
.wave-bar:nth-child(8) { animation-delay: 0.15s; }
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "pending_confirmation" not in st.session_state:
    st.session_state.pending_confirmation = None

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding: 8px 0 20px;'>
        <div style='font-family: Space Mono, monospace; font-size: 22px; font-weight: 700; color: #7c6af7;'>ARIA</div>
        <div style='font-size: 12px; color: #6b6b8a; margin-top: 2px;'>Autonomous Reasoning & Interaction Agent</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**Model Settings**")
    stt_provider = st.selectbox("STT Provider", ["groq", "openai", "whisper_local"], index=0,
                                 help="groq/openai requires API key; whisper_local runs offline")
    llm_provider = st.selectbox("LLM Provider", ["ollama", "groq", "openai"], index=0,
                                 help="ollama = fully local; groq/openai = API-based")
    llm_model = st.text_input("LLM Model", value="llama3.2" if llm_provider == "ollama" else "llama-3.1-8b-instant",
                               help="e.g. llama3.2, mistral, gemma2 for Ollama")

    st.markdown("---")
    st.markdown("**API Keys** *(if using API providers)*")
    groq_key = st.text_input("Groq API Key", type="password", value=os.getenv("GROQ_API_KEY", ""))
    openai_key = st.text_input("OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY", ""))

    if groq_key:
        os.environ["GROQ_API_KEY"] = groq_key
    if openai_key:
        os.environ["OPENAI_API_KEY"] = openai_key

    st.markdown("---")
    st.markdown("**Options**")
    human_in_loop = st.toggle("Confirm before file ops", value=True,
                               help="Prompts you before creating/writing files")
    show_raw = st.toggle("Show raw intent JSON", value=False)

    st.markdown("---")
    st.markdown("**Supported Intents**")
    intents = ["📄 Create file", "💻 Write code", "📝 Summarize text", "💬 General chat", "🔗 Compound commands"]
    for i in intents:
        st.markdown(f"<div style='font-size:12px; color:#9090b0; padding:2px 0;'>{i}</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**Session History**")
    if not st.session_state.history:
        st.markdown("<div style='font-size:12px; color:#6b6b8a;'>No actions yet.</div>", unsafe_allow_html=True)
    for i, item in enumerate(reversed(st.session_state.history[-10:])):
        st.markdown(f"""
        <div class='history-item'>
            <div class='history-ts'>{item['timestamp']}</div>
            <div class='history-text'>{item['transcription'][:60]}…</div>
            <span class='tag tag-intent'>{item['primary_intent']}</span>
        </div>
        """, unsafe_allow_html=True)

# ── Main layout ───────────────────────────────────────────────────────────────
col_title, col_status = st.columns([3, 1])
with col_title:
    st.markdown("""
    <h1 style='margin-bottom: 4px; font-size: 28px;'>🎙️ Voice AI Agent</h1>
    <p style='color: #6b6b8a; font-size: 14px; margin-top: 0;'>Speak a command → transcribe → understand intent → execute locally</p>
    """, unsafe_allow_html=True)
with col_status:
    st.markdown(f"""
    <div style='text-align:right; padding-top: 14px;'>
        <span class='tag tag-success'>● READY</span>
        <div style='font-family: Space Mono, monospace; font-size: 10px; color: #6b6b8a; margin-top: 4px;'>{llm_provider}/{llm_model}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ── Input tabs ────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🎤  Microphone", "📁  Upload Audio", "⌨️  Text (Debug)"])

audio_bytes = None
input_mode = None

with tab1:
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    mic_audio = st.audio_input("Click to record your command")
    if mic_audio:
        audio_bytes = mic_audio.read()
        input_mode = "mic"

with tab2:
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload a WAV or MP3 file", type=["wav", "mp3", "m4a", "ogg", "flac"])
    if uploaded:
        audio_bytes = uploaded.read()
        input_mode = "upload"
        st.audio(uploaded)

with tab3:
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:12px; color:#fbbf24; margin-bottom:8px;'>⚡ Debug mode: bypass STT, type command directly</div>", unsafe_allow_html=True)
    debug_text = st.text_area("Type your command", placeholder="e.g. Create a Python file with a bubble sort function", height=80)
    if st.button("Run Command"):
        input_mode = "text"
        audio_bytes = None

st.markdown("---")

# ── Pending confirmation ───────────────────────────────────────────────────────
if st.session_state.pending_confirmation:
    pending = st.session_state.pending_confirmation
    st.markdown(f"""
    <div class='card card-warn'>
        <div style='font-family: Space Mono, monospace; font-size: 13px; color: #fbbf24; margin-bottom: 8px;'>⚠ CONFIRM ACTION</div>
        <div style='font-size: 13px; color: #e2e2f0; margin-bottom: 12px;'>{pending['description']}</div>
        <div style='font-size: 12px; color: #6b6b8a;'>This will write to: <code style='color:#c084fc'>output/{pending.get('filename','...')}</code></div>
    </div>
    """, unsafe_allow_html=True)
    c1, c2 = st.columns([1, 4])
    with c1:
        if st.button("✓ Confirm"):
            result = execute_tool(pending["intent_data"], confirmed=True)
            st.session_state.last_result = result
            st.session_state.pending_confirmation = None
            st.rerun()
    with c2:
        if st.button("✗ Cancel"):
            st.session_state.pending_confirmation = None
            st.session_state.last_result = {"status": "cancelled", "message": "Action cancelled by user."}
            st.rerun()

# ── Processing ────────────────────────────────────────────────────────────────
should_process = (audio_bytes is not None and input_mode in ("mic", "upload")) or \
                 (input_mode == "text" and debug_text.strip())

if should_process and not st.session_state.pending_confirmation:
    with st.spinner(""):
        pipeline_placeholder = st.empty()

        # Step 1: STT
        if input_mode == "text":
            transcription = debug_text.strip()
            stt_time = 0
        else:
            pipeline_placeholder.markdown("""
            <div class='card'>
                <div class='waveform'>
                    <div class='wave-bar'></div><div class='wave-bar'></div><div class='wave-bar'></div>
                    <div class='wave-bar'></div><div class='wave-bar'></div><div class='wave-bar'></div>
                    <div class='wave-bar'></div><div class='wave-bar'></div>
                    <span style='font-family: Space Mono, monospace; font-size: 12px; color: #7c6af7; margin-left: 12px;'>Transcribing audio…</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            t0 = time.time()
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name
            try:
                transcription = transcribe_audio(tmp_path, provider=stt_provider)
            except Exception as e:
                transcription = f"[STT ERROR: {e}]"
            finally:
                os.unlink(tmp_path)
            stt_time = round(time.time() - t0, 2)

        if not transcription or transcription.startswith("[STT ERROR"):
            st.markdown(f"""
            <div class='card card-err'>
                <b>STT Failed</b><br>
                <span style='font-family: Space Mono, monospace; font-size: 12px;'>{transcription}</span>
            </div>
            """, unsafe_allow_html=True)
            st.stop()

        # Step 2: Intent classification
        pipeline_placeholder.markdown("""
        <div class='card'>
            <span style='font-family: Space Mono, monospace; font-size: 12px; color: #7c6af7;'>🧠 Classifying intent…</span>
        </div>
        """, unsafe_allow_html=True)
        t1 = time.time()
        try:
            intent_data = classify_intent(transcription, provider=llm_provider, model=llm_model)
        except Exception as e:
            intent_data = {"primary_intent": "general_chat", "entities": {}, "error": str(e)}
        intent_time = round(time.time() - t1, 2)

        # Step 3: Tool execution
        pipeline_placeholder.markdown("""
        <div class='card'>
            <span style='font-family: Space Mono, monospace; font-size: 12px; color: #7c6af7;'>⚙️ Executing action…</span>
        </div>
        """, unsafe_allow_html=True)

        needs_confirm = human_in_loop and intent_data.get("primary_intent") in ("create_file", "write_code")
        if needs_confirm:
            fname = intent_data.get("entities", {}).get("filename", "file.txt")
            st.session_state.pending_confirmation = {
                "description": f"About to execute: **{intent_data.get('primary_intent')}** — {intent_data.get('entities', {}).get('description', '')}",
                "filename": fname,
                "intent_data": intent_data,
            }
            pipeline_placeholder.empty()
            # Store partial result for display
            st.session_state.last_result = {
                "status": "awaiting_confirmation",
                "transcription": transcription,
                "intent_data": intent_data,
                "stt_time": stt_time,
                "intent_time": intent_time,
            }
            st.rerun()

        t2 = time.time()
        try:
            result = execute_tool(intent_data)
        except Exception as e:
            result = {"status": "error", "message": str(e), "output": ""}
        exec_time = round(time.time() - t2, 2)

        pipeline_placeholder.empty()
        result["transcription"] = transcription
        result["intent_data"] = intent_data
        result["stt_time"] = stt_time
        result["intent_time"] = intent_time
        result["exec_time"] = exec_time

        # Save to history
        st.session_state.history.append({
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "transcription": transcription,
            "primary_intent": intent_data.get("primary_intent", "unknown"),
            "status": result.get("status", "unknown"),
        })
        st.session_state.last_result = result

# ── Results display ───────────────────────────────────────────────────────────
if st.session_state.last_result:
    r = st.session_state.last_result
    if r.get("status") == "awaiting_confirmation":
        pass  # confirmation UI already shown above
    else:
        st.markdown("## Pipeline Results")
        col_a, col_b = st.columns([1, 1])

        with col_a:
            intent_data = r.get("intent_data", {})
            primary = intent_data.get("primary_intent", "unknown")
            entities = intent_data.get("entities", {})
            compound = intent_data.get("compound_intents", [])

            st.markdown(f"""
            <div class='card card-accent'>
                <div style='font-size: 11px; color: #6b6b8a; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 12px;'>Pipeline Trace</div>
                <div class='step'>
                    <div class='step-num'>01</div>
                    <div class='step-content'>
                        <div class='step-label'>Transcription {'(' + str(r.get('stt_time','')) + 's)' if r.get('stt_time') else ''}</div>
                        <div class='step-value'>"{r.get('transcription','')}"</div>
                    </div>
                </div>
                <div class='step'>
                    <div class='step-num'>02</div>
                    <div class='step-content'>
                        <div class='step-label'>Detected Intent {'(' + str(r.get('intent_time','')) + 's)' if r.get('intent_time') else ''}</div>
                        <div class='step-value'>
                            <span class='tag tag-intent'>{primary}</span>
                            {''.join([f"<span class='tag tag-warn' style='margin-left:4px'>{c}</span>" for c in compound]) if compound else ''}
                        </div>
                    </div>
                </div>
                <div class='step'>
                    <div class='step-num'>03</div>
                    <div class='step-content'>
                        <div class='step-label'>Action Executed {'(' + str(r.get('exec_time','')) + 's)' if r.get('exec_time') else ''}</div>
                        <div class='step-value'>{r.get('action_taken', primary)}</div>
                    </div>
                </div>
                <div class='step'>
                    <div class='step-num'>04</div>
                    <div class='step-content'>
                        <div class='step-label'>Status</div>
                        <div class='step-value'>
                            <span class='tag {"tag-success" if r.get("status") == "success" else "tag-warn"}'>{r.get('status','unknown').upper()}</span>
                            <span style='font-size:12px; color:#9090b0; margin-left:8px;'>{r.get('message','')}</span>
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            if entities and show_raw:
                st.markdown(f"<div class='code-out'>{json.dumps(intent_data, indent=2)}</div>", unsafe_allow_html=True)

        with col_b:
            output = r.get("output", "")
            files = r.get("files_created", [])

            st.markdown(f"""
            <div class='card {"card-success" if r.get("status") == "success" else "card-err"}'>
                <div style='font-size: 11px; color: #6b6b8a; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 10px;'>Output</div>
            """, unsafe_allow_html=True)

            if files:
                for f in files:
                    st.markdown(f"<div style='font-family: Space Mono, monospace; font-size: 11px; color: #34d399;'>📄 output/{f}</div>", unsafe_allow_html=True)
                st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

            if output:
                st.markdown(f"<div class='code-out'>{output[:3000]}{'…' if len(output) > 3000 else ''}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div style='font-size:13px; color:#9090b0;'>{r.get('message', 'No output.')}</div>", unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)

        # Download buttons for created files
        if files:
            st.markdown("**Download created files:**")
            for fname in files:
                fpath = OUTPUT_DIR / fname
                if fpath.exists():
                    st.download_button(
                        label=f"⬇ {fname}",
                        data=fpath.read_text(errors="replace"),
                        file_name=fname,
                        mime="text/plain",
                        key=f"dl_{fname}_{time.time()}"
                    )
