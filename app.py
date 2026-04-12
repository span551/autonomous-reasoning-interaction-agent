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

st.set_page_config(
    page_title="ARIA – Voice AI Agent",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:ital,wght@0,400;0,700;1,400&family=DM+Sans:wght@300;400;500;600&display=swap');
:root {
    --bg:#0a0a0f;--surface:#111118;--surface2:#1a1a24;--accent:#7c6af7;
    --accent2:#c084fc;--success:#34d399;--warn:#fbbf24;--err:#f87171;
    --text:#e2e2f0;--muted:#6b6b8a;--border:#2a2a3a;
}
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;}
.stApp{background:var(--bg);color:var(--text);}
[data-testid="stSidebar"]{background:var(--surface);border-right:1px solid var(--border);}
h1,h2,h3{font-family:'Space Mono',monospace;}
.card{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:20px 24px;margin:10px 0;}
.card-accent{border-left:3px solid var(--accent);}
.card-success{border-left:3px solid var(--success);}
.card-warn{border-left:3px solid var(--warn);}
.card-err{border-left:3px solid var(--err);}
.tag{display:inline-block;padding:3px 12px;border-radius:20px;font-family:'Space Mono',monospace;font-size:11px;font-weight:700;letter-spacing:.05em;text-transform:uppercase;}
.tag-intent{background:#2d2060;color:var(--accent2);border:1px solid #4a3a90;}
.tag-success{background:#0d2e22;color:var(--success);border:1px solid #1a5a3a;}
.tag-warn{background:#2e2100;color:var(--warn);border:1px solid #5a3e00;}
.step{display:flex;align-items:flex-start;gap:16px;padding:14px 0;border-bottom:1px solid var(--border);}
.step:last-child{border-bottom:none;}
.step-num{font-family:'Space Mono',monospace;font-size:11px;color:var(--accent);background:#1e1840;border:1px solid #3d2e80;border-radius:4px;padding:2px 8px;white-space:nowrap;margin-top:2px;}
.step-content{flex:1;}
.step-label{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.08em;margin-bottom:4px;}
.step-value{font-family:'Space Mono',monospace;font-size:13px;color:var(--text);word-break:break-word;}
.history-item{background:var(--surface2);border:1px solid var(--border);border-radius:8px;padding:12px 16px;margin:6px 0;}
.history-ts{font-family:'Space Mono',monospace;font-size:10px;color:var(--muted);}
.history-text{font-size:13px;color:var(--text);margin:4px 0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.code-out{background:#0d0d14;border:1px solid var(--border);border-radius:8px;padding:16px;font-family:'Space Mono',monospace;font-size:12px;line-height:1.6;color:#b4b4d4;overflow-x:auto;white-space:pre-wrap;word-break:break-word;}
.groq-badge{display:inline-flex;align-items:center;gap:6px;background:#1a1040;border:1px solid #3d2e80;border-radius:8px;padding:6px 12px;font-family:'Space Mono',monospace;font-size:11px;color:var(--accent2);}
.stButton>button{font-family:'Space Mono',monospace!important;font-size:12px!important;letter-spacing:.05em!important;background:var(--accent)!important;color:white!important;border:none!important;border-radius:8px!important;padding:8px 20px!important;transition:opacity .2s!important;}
.stButton>button:hover{opacity:.85!important;}
/* Red cancel button */
.cancel-btn>button{background:#7f1d1d!important;color:#fca5a5!important;}
hr{border-color:var(--border);margin:24px 0;}
/* Confirm modal overlay */
.confirm-wrap{background:#111118;border:1px solid #5a3e00;border-radius:12px;padding:24px;margin:24px 0;}
.confirm-title{font-family:'Space Mono',monospace;font-size:13px;color:#fbbf24;margin-bottom:12px;}
.confirm-body{font-size:13px;color:#e2e2f0;margin-bottom:16px;line-height:1.6;}
.confirm-path{font-family:'Space Mono',monospace;font-size:12px;color:#c084fc;background:#1a0a30;border:1px solid #3d2060;border-radius:6px;padding:6px 12px;display:inline-block;margin-top:4px;}
@keyframes wave{0%,100%{height:4px;}50%{height:20px;}}
.waveform{display:flex;align-items:center;gap:3px;height:28px;}
.wave-bar{width:3px;background:var(--accent);border-radius:2px;animation:wave 1.2s ease-in-out infinite;}
.wave-bar:nth-child(2){animation-delay:.1s;}.wave-bar:nth-child(3){animation-delay:.2s;}
.wave-bar:nth-child(4){animation-delay:.3s;}.wave-bar:nth-child(5){animation-delay:.4s;}
.wave-bar:nth-child(6){animation-delay:.5s;}.wave-bar:nth-child(7){animation-delay:.3s;}
.wave-bar:nth-child(8){animation-delay:.15s;}
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────────
for key, default in [
    ("history", []),
    ("last_result", None),
    ("pending_confirmation", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:8px 0 20px;'>
        <div style='font-family:Space Mono,monospace;font-size:22px;font-weight:700;color:#7c6af7;'>ARIA</div>
        <div style='font-size:12px;color:#6b6b8a;margin-top:2px;'>Autonomous Reasoning & Interaction Agent</div>
    </div>
    <div style='margin-bottom:16px;'><div class='groq-badge'>⚡ Powered by Groq API</div></div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**LLM Model**")
    llm_model = st.selectbox(
        "model",
        ["llama-3.1-8b-instant", "llama-3.3-70b-versatile", "mixtral-8x7b-32768", "gemma2-9b-it"],
        index=0,
        help="llama-3.1-8b-instant = fastest; 70b = most capable",
        label_visibility="collapsed"
    )
    st.markdown("<div style='font-size:11px;color:#6b6b8a;'>STT: whisper-large-v3 (fixed)</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**Options**")
    human_in_loop = st.toggle("Confirm before file ops", value=True)
    show_raw      = st.toggle("Show raw intent JSON",    value=False)

    st.markdown("---")
    st.markdown("**Supported Intents**")
    for item in ["📄 Create file", "💻 Write code", "📝 Summarize text",
                 "💬 General chat", "📁 Create folder", "🔗 Compound commands"]:
        st.markdown(f"<div style='font-size:12px;color:#9090b0;padding:2px 0;'>{item}</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**Session History**")
    if not st.session_state.history:
        st.markdown("<div style='font-size:12px;color:#6b6b8a;'>No actions yet.</div>", unsafe_allow_html=True)
    for item in reversed(st.session_state.history[-10:]):
        st.markdown(f"""
        <div class='history-item'>
            <div class='history-ts'>{item['timestamp']}</div>
            <div class='history-text'>{item['transcription'][:60]}…</div>
            <span class='tag tag-intent'>{item['primary_intent']}</span>
        </div>""", unsafe_allow_html=True)

# ── Main header ────────────────────────────────────────────────────────────────
col_title, col_status = st.columns([3, 1])
with col_title:
    st.markdown("""
    <h1 style='margin-bottom:4px;font-size:28px;'>🎙️ Voice AI Agent</h1>
    <p style='color:#6b6b8a;font-size:14px;margin-top:0;'>Speak a command → transcribe → understand intent → execute locally</p>
    """, unsafe_allow_html=True)
with col_status:
    st.markdown(f"""
    <div style='text-align:right;padding-top:14px;'>
        <span class='tag tag-success'>● READY</span>
        <div style='font-family:Space Mono,monospace;font-size:10px;color:#6b6b8a;margin-top:4px;'>groq / {llm_model}</div>
        <div style='font-family:Space Mono,monospace;font-size:10px;color:#6b6b8a;'>whisper-large-v3</div>
    </div>""", unsafe_allow_html=True)

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# CONFIRM DIALOG — shown instead of input tabs when pending
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.pending_confirmation:
    pending = st.session_state.pending_confirmation
    intent  = pending["intent_data"]
    fname   = intent.get("entities", {}).get("filename", "file.txt")
    action  = intent.get("primary_intent", "write_code").replace("_", " ").title()
    desc    = intent.get("entities", {}).get("description", "")
    transcription = pending.get("transcription", "")

    st.markdown(f"""
    <div class='confirm-wrap'>
        <div class='confirm-title'>⚠ Confirm File Operation</div>
        <div class='confirm-body'>
            You said: <span style='color:#c084fc;font-style:italic;'>"{transcription}"</span><br><br>
            ARIA wants to <b>{action}</b>{(' — ' + desc) if desc else ''}.<br>
            This will create or overwrite:
        </div>
        <div class='confirm-path'>output/{fname}</div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, _ = st.columns([1, 1, 3])
    with c1:
        if st.button("✓  Run it", use_container_width=True):
            with st.spinner("Executing…"):
                result = execute_tool(pending["intent_data"], confirmed=True)
            result.update({
                "transcription": pending.get("transcription", ""),
                "intent_data":   pending["intent_data"],
                "stt_time":      pending.get("stt_time", 0),
                "intent_time":   pending.get("intent_time", 0),
                "exec_time":     0,
            })
            st.session_state.history.append({
                "timestamp":      datetime.now().strftime("%H:%M:%S"),
                "transcription":  pending.get("transcription", ""),
                "primary_intent": intent.get("primary_intent", "unknown"),
                "status":         result.get("status", "unknown"),
            })
            st.session_state.last_result = result
            st.session_state.pending_confirmation = None
            st.rerun()
    with c2:
        if st.button("✗  Cancel", use_container_width=True):
            st.session_state.pending_confirmation = None
            st.session_state.last_result = {
                "status": "cancelled",
                "message": "Action cancelled.",
                "transcription": pending.get("transcription", ""),
                "intent_data": pending["intent_data"],
                "output": "",
                "action_taken": "Cancelled by user",
                "files_created": [],
            }
            st.rerun()

    st.stop()   # Don't render input tabs while confirm is shown

# ── Input tabs ─────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🎤  Microphone", "📁  Upload Audio", "⌨️  Text (Debug)"])
audio_bytes = None
input_mode  = None
debug_text  = ""

with tab1:
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    mic_audio = st.audio_input("Click to record your command")
    if mic_audio:
        audio_bytes = mic_audio.read()
        input_mode  = "mic"

with tab2:
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload a WAV or MP3 file", type=["wav", "mp3", "m4a", "ogg", "flac"])
    if uploaded:
        audio_bytes = uploaded.read()
        input_mode  = "upload"
        st.audio(uploaded)

with tab3:
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:12px;color:#fbbf24;margin-bottom:8px;'>⚡ Debug mode — bypass STT and type a command directly</div>", unsafe_allow_html=True)
    debug_text = st.text_area("Command", placeholder="e.g. Create a Python file with a bubble sort function", height=80, label_visibility="collapsed")
    if st.button("Run Command"):
        if debug_text.strip():
            input_mode = "text"

st.markdown("---")

# ── Processing pipeline ────────────────────────────────────────────────────────
should_process = (audio_bytes is not None and input_mode in ("mic", "upload")) or \
                 (input_mode == "text" and debug_text.strip())

if should_process:
    with st.spinner(""):
        ph = st.empty()

        # 1 — STT
        if input_mode == "text":
            transcription = debug_text.strip()
            stt_time = 0
        else:
            ph.markdown("""<div class='card'><div class='waveform'>
                <div class='wave-bar'></div><div class='wave-bar'></div><div class='wave-bar'></div>
                <div class='wave-bar'></div><div class='wave-bar'></div><div class='wave-bar'></div>
                <div class='wave-bar'></div><div class='wave-bar'></div>
                <span style='font-family:Space Mono,monospace;font-size:12px;color:#7c6af7;margin-left:12px;'>Transcribing with Groq Whisper…</span>
            </div></div>""", unsafe_allow_html=True)
            t0 = time.time()
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name
            try:
                transcription = transcribe_audio(tmp_path)
            except Exception as e:
                transcription = f"[STT ERROR: {e}]"
            finally:
                os.unlink(tmp_path)
            stt_time = round(time.time() - t0, 2)

        if not transcription or transcription.startswith("[STT ERROR"):
            ph.markdown(f"<div class='card card-err'><b>Transcription Failed</b><br><span style='font-size:12px;'>{transcription}</span></div>", unsafe_allow_html=True)
            st.stop()

        # 2 — Intent
        ph.markdown("<div class='card'><span style='font-family:Space Mono,monospace;font-size:12px;color:#7c6af7;'>🧠 Classifying intent via Groq…</span></div>", unsafe_allow_html=True)
        t1 = time.time()
        try:
            intent_data = classify_intent(transcription, model=llm_model)
        except Exception as e:
            intent_data = {"primary_intent": "general_chat", "entities": {}, "error": str(e), "original_text": transcription}
        intent_time = round(time.time() - t1, 2)

        # 3 — Confirm gate or execute
        ph.empty()
        needs_confirm = human_in_loop and intent_data.get("primary_intent") in ("create_file", "write_code")

        if needs_confirm:
            # Store everything needed and rerun — confirm dialog renders at top of screen
            st.session_state.pending_confirmation = {
                "intent_data":   intent_data,
                "transcription": transcription,
                "stt_time":      stt_time,
                "intent_time":   intent_time,
            }
            st.session_state.last_result = None  # clear stale results
            st.rerun()

        ph.markdown("<div class='card'><span style='font-family:Space Mono,monospace;font-size:12px;color:#7c6af7;'>⚙️ Executing…</span></div>", unsafe_allow_html=True)
        t2 = time.time()
        try:
            result = execute_tool(intent_data)
        except Exception as e:
            result = {"status": "error", "message": str(e), "output": "", "action_taken": "Error", "files_created": []}
        exec_time = round(time.time() - t2, 2)

        ph.empty()
        result.update({
            "transcription": transcription,
            "intent_data":   intent_data,
            "stt_time":      stt_time,
            "intent_time":   intent_time,
            "exec_time":     exec_time,
        })
        st.session_state.history.append({
            "timestamp":      datetime.now().strftime("%H:%M:%S"),
            "transcription":  transcription,
            "primary_intent": intent_data.get("primary_intent", "unknown"),
            "status":         result.get("status", "unknown"),
        })
        st.session_state.last_result = result

# ── Results ────────────────────────────────────────────────────────────────────
if st.session_state.last_result:
    r = st.session_state.last_result
    st.markdown("## Pipeline Results")
    col_a, col_b = st.columns(2)

    with col_a:
        id_      = r.get("intent_data", {})
        primary  = id_.get("primary_intent", "unknown")
        compound = id_.get("compound_intents", [])

        # Pre-build every dynamic fragment so no logic lives inside the HTML string
        stt_label    = "TRANSCRIPTION (" + str(r["stt_time"]) + "s)" if r.get("stt_time") else "TRANSCRIPTION"
        intent_label = "DETECTED INTENT (" + str(r["intent_time"]) + "s)" if r.get("intent_time") else "DETECTED INTENT"
        exec_label   = "ACTION EXECUTED (" + str(r["exec_time"]) + "s)" if r.get("exec_time") else "ACTION EXECUTED"
        transcription_text = r.get("transcription", "")
        action_text  = r.get("action_taken", primary)
        message_text = r.get("message", "")
        status_val   = r.get("status", "unknown").upper()
        status_class = "tag-success" if r.get("status") == "success" else "tag-warn"
        compound_tags = "".join(
            "<span style='display:inline-block;padding:3px 12px;border-radius:20px;"
            "font-family:Space Mono,monospace;font-size:11px;font-weight:700;"
            "background:#2e2100;color:#fbbf24;border:1px solid #5a3e00;margin-left:4px;'>"
            + c + "</span>" for c in compound
        )

        st.markdown("""
        <div class="card card-accent">
            <div style="font-size:11px;color:#6b6b8a;text-transform:uppercase;letter-spacing:.08em;margin-bottom:12px;">Pipeline Trace</div>
            <div class="step">
                <div class="step-num">01</div>
                <div class="step-content">
                    <div class="step-label">""" + stt_label + """</div>
                    <div class="step-value">&ldquo;""" + transcription_text + """&rdquo;</div>
                </div>
            </div>
            <div class="step">
                <div class="step-num">02</div>
                <div class="step-content">
                    <div class="step-label">""" + intent_label + """</div>
                    <div class="step-value">
                        <span class="tag tag-intent">""" + primary + """</span>
                        """ + compound_tags + """
                    </div>
                </div>
            </div>
            <div class="step">
                <div class="step-num">03</div>
                <div class="step-content">
                    <div class="step-label">""" + exec_label + """</div>
                    <div class="step-value">""" + action_text + """</div>
                </div>
            </div>
            <div class="step">
                <div class="step-num">04</div>
                <div class="step-content">
                    <div class="step-label">STATUS</div>
                    <div class="step-value">
                        <span class="tag """ + status_class + """">""" + status_val + """</span>
                        <span style="font-size:12px;color:#9090b0;margin-left:8px;">""" + message_text + """</span>
                    </div>
                </div>
            </div>
        </div>""", unsafe_allow_html=True)

        if show_raw and id_:
            raw_json = json.dumps(id_, indent=2)
            st.markdown("<div class='code-out'>" + raw_json + "</div>", unsafe_allow_html=True)

    with col_b:
        output = r.get("output", "")
        files  = r.get("files_created", [])
        card_class = "card-success" if r.get("status") == "success" else "card-err"
        st.markdown(
            "<div class='card " + card_class + "'>"
            "<div style='font-size:11px;color:#6b6b8a;text-transform:uppercase;"
            "letter-spacing:.08em;margin-bottom:10px;'>Output</div>",
            unsafe_allow_html=True
        )
        for f in files:
            st.markdown(
                "<div style='font-family:Space Mono,monospace;font-size:11px;"
                "color:#34d399;margin-bottom:4px;'>📄 output/" + f + "</div>",
                unsafe_allow_html=True
            )
        if files:
            st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
        if output:
            clipped = output[:3000] + ("…" if len(output) > 3000 else "")
            st.markdown("<div class='code-out'>" + clipped + "</div>", unsafe_allow_html=True)
        else:
            fallback = r.get("message", "No output.")
            st.markdown(
                "<div style='font-size:13px;color:#9090b0;'>" + fallback + "</div>",
                unsafe_allow_html=True
            )
        st.markdown("</div>", unsafe_allow_html=True)

    if files:
        st.markdown("**Download created files:**")
        for fname in files:
            fpath = OUTPUT_DIR / fname
            if fpath.exists():
                st.download_button(
                    f"⬇  {fname}",
                    data=fpath.read_text(errors="replace"),
                    file_name=fname,
                    mime="text/plain",
                    key=f"dl_{fname}_{time.time()}"
                )
