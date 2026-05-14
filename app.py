"""
app.py — Streamlit UI for Deep Agent Coder

Run with:  streamlit run app.py
"""

import asyncio
import base64
import json
import os
from io import StringIO
from pathlib import Path
from typing import Optional
import docx as _docx #type: ignore
import pandas as pd #type: ignore
import streamlit as st #type: ignore
import plotly.express as px #type: ignore
from dotenv import load_dotenv
import pdfplumber #type: ignore
load_dotenv()

from formatter import format_message_content
from agent.deep_agent import agent

# ─────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Deep Agent Coder",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────



# ─────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────
def _init_state():
    defaults = {
        "lc_messages": [],      # Full LangChain message list (passed to agent)
        "display_turns": [],    # List of turn dicts for display
        "file_contexts": {},    # filename → text content (for display/context)
        "files_raw": {},        # filename → raw file content (for sandbox access)
        "dataframes": {},       # filename → pd.DataFrame
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ─────────────────────────────────────────────────────────────
# File processing
# ─────────────────────────────────────────────────────────────
def process_file(uploaded_file) -> tuple[str, Optional[pd.DataFrame], Optional[str]]:
    """Return (text_content, dataframe_or_None, raw_content_or_None) for any uploaded file type."""
    name = uploaded_file.name
    ext = Path(name).suffix.lower()

    if ext in (".csv",):
        # Read raw CSV content first
        raw_content = uploaded_file.read().decode("utf-8", errors="replace")
        uploaded_file.seek(0)  # Reset for pandas read
        
        df = pd.read_csv(uploaded_file)
        snippet = df.head(20).to_string()
        content = (
            f"CSV file '{name}' — {len(df)} rows × {len(df.columns)} columns.\n"
            f"Columns: {', '.join(df.columns.tolist())}\n\n"
            f"First 20 rows:\n{snippet}"
        )
        return content, df, raw_content

    elif ext in (".xlsx", ".xls"):
        df = pd.read_excel(uploaded_file)
        snippet = df.head(20).to_string()
        content = (
            f"Excel file '{name}' — {len(df)} rows × {len(df.columns)} columns.\n"
            f"Columns: {', '.join(df.columns.tolist())}\n\n"
            f"First 20 rows:\n{snippet}"
        )
        return content, df, None

    elif ext in (".txt", ".md", ".py", ".js", ".ts", ".json", ".yaml", ".yml", ".toml", ".sh"):
        raw = uploaded_file.read().decode("utf-8", errors="replace")
        return f"File '{name}':\n```\n{raw[:8000]}\n```", None, raw

    elif ext == ".pdf":
        try:
            with pdfplumber.open(uploaded_file) as pdf:
                pages = [p.extract_text() or "" for p in pdf.pages]
            text = "\n\n".join(pages)[:12000]
            return f"PDF '{name}' ({len(pages)} pages):\n{text}", None, None
        except ImportError:
            return f"[PDF '{name}': install pdfplumber to extract text]", None, None

    elif ext in (".docx",):
        try:
            doc = _docx.Document(uploaded_file)
            text = "\n".join(p.text for p in doc.paragraphs)[:12000]
            return f"Word document '{name}':\n{text}", None, None
        except ImportError:
            return f"[DOCX '{name}': install python-docx to extract text]", None, None

    elif ext in (".png", ".jpg", ".jpeg", ".gif", ".webp"):
        data = uploaded_file.read()
        b64 = base64.b64encode(data).decode()
        mime = "image/png" if ext == ".png" else "image/jpeg"
        return f"[Image '{name}' ({len(data)//1024} KB) — passed as base64]", None, None

    else:
        try:
            raw = uploaded_file.read().decode("utf-8", errors="replace")
            return f"File '{name}':\n{raw[:6000]}", None, raw
        except Exception:
            return f"[Binary file '{name}' — cannot display as text]", None, None


def build_user_content(prompt: str) -> str:
    """Inject uploaded file context into the user's prompt."""
    if not st.session_state.file_contexts:
        return prompt

    parts = []
    for fname, content in st.session_state.file_contexts.items():
        parts.append(f"=== Uploaded file: {fname} ===\n{content}\n")

    context_block = "\n".join(parts)
    return (
        f"The user has uploaded the following file(s):\n\n"
        f"{context_block}\n"
        f"{'─'*60}\n\n"
        f"User request: {prompt}"
    )


# ─────────────────────────────────────────────────────────────
# Agent invocation
# ─────────────────────────────────────────────────────────────
async def _invoke_agent(user_content: str):
    """Add user message to history and invoke agent with full conversation."""
    # Build the new user message
    new_msg = {"role": "user", "content": user_content}

    # Pass full conversation history + new message
    all_messages = st.session_state.lc_messages + [new_msg]
    
    # Build initial state with files available to the agent
    initial_state = {
        "messages": all_messages,
        "files": st.session_state.files_raw,  # Pass raw file contents to agent state
    }

    result = await agent.ainvoke(initial_state)
    return result


# ─────────────────────────────────────────────────────────────
# CSV explorer widget
# ─────────────────────────────────────────────────────────────
def render_csv_explorer(fname: str, df: pd.DataFrame):
    with st.expander(f"📊  `{fname}`  —  {len(df)} rows × {len(df.columns)} cols", expanded=True):
        tab_data, tab_stats, tab_viz = st.tabs(["Data", "Statistics", "Visualize"])

        with tab_data:
            st.dataframe(df, use_container_width='stretch', height=300)

        with tab_stats:
            desc = df.describe(include="all")
            st.dataframe(desc, use_container_width='stretch')

        with tab_viz:
            num_cols = df.select_dtypes(include="number").columns.tolist()
            all_cols = df.columns.tolist()

            if not num_cols:
                st.info("No numeric columns detected for charting.")
                return

            c1, c2 = st.columns([1, 3])
            chart_type = c1.radio(
                "Chart", ["Line", "Bar", "Scatter", "Histogram", "Area"],
                key=f"ct_{fname}"
            )

            with c2:
                if chart_type == "Scatter":
                    x = st.selectbox("X", all_cols, key=f"sx_{fname}")
                    y = st.selectbox("Y", num_cols, key=f"sy_{fname}")
                    try:
                        st.plotly_chart(px.scatter(df, x=x, y=y, title=f"{x} vs {y}"),
                                        use_container_width='stretch')
                    except ImportError:
                        st.scatter_chart(df.set_index(x)[[y]])

                elif chart_type == "Histogram":
                    col = st.selectbox("Column", num_cols, key=f"hc_{fname}")
                    try:
                        st.plotly_chart(px.histogram(df, x=col), use_container_width='stretch')
                    except ImportError:
                        st.bar_chart(df[col].value_counts())

                elif chart_type == "Bar":
                    cols = st.multiselect("Columns", num_cols,
                                          default=num_cols[:1], key=f"bc_{fname}")
                    if cols:
                        st.bar_chart(df[cols])

                elif chart_type == "Area":
                    cols = st.multiselect("Columns", num_cols,
                                          default=num_cols[:2], key=f"ac_{fname}")
                    if cols:
                        st.area_chart(df[cols])

                else:  # Line
                    cols = st.multiselect("Columns", num_cols,
                                          default=num_cols[:2], key=f"lc_{fname}")
                    if cols:
                        st.line_chart(df[cols])


# ─────────────────────────────────────────────────────────────
# Message rendering
# ─────────────────────────────────────────────────────────────
def render_turn(turn: dict):
    """Render one agent turn: user message + all agent/tool messages."""

    # User
    with st.chat_message("user", avatar="🧑"):
        # Show the clean prompt (without the injected file context blob)
        st.markdown(turn.get("user_display", ""))

    # Agent messages (AI + tool calls/results interleaved)
    for msg in turn.get("agent_messages", []):
        _render_one_message(msg)


def _render_one_message(msg):
    msg_type = msg.__class__.__name__

    if "Ai" in msg_type or msg_type == "AIMessage":
        content = ""
        if isinstance(msg.content, str):
            content = msg.content
        elif isinstance(msg.content, list):
            content = " ".join(
                item.get("text", "") for item in msg.content
                if isinstance(item, dict) and item.get("type") == "text"
            )

        has_tool_calls = hasattr(msg, "tool_calls") and msg.tool_calls

        with st.chat_message("assistant", avatar="⚡"):
            if content.strip():
                st.markdown(content)
            if has_tool_calls:
                for tc in msg.tool_calls:
                    tool_name = tc.get("name", "tool")
                    args = tc.get("args", {})
                    # Special rendering for code execution
                    if tool_name == "run_code_in_sandbox":
                        with st.expander(f"🐳 Sandbox: executing code", expanded=False):
                            code = args.get("code", "")
                            st.code(code, language="python")
                    else:
                        with st.expander(f"🔧 `{tool_name}`", expanded=False):
                            st.code(json.dumps(args, indent=2, ensure_ascii=False),
                                    language="json")

    elif "Tool" in msg_type:
        tool_name = getattr(msg, "name", "tool_result")
        raw = format_message_content(msg)

        # Try to detect code execution results
        is_sandbox = "Execution" in raw or "User-" in raw or "ERROR:" in raw or "[IMAGE]" in raw

        if is_sandbox:
            label = "✅ Sandbox output" if "ERROR:" not in raw else "❌ Sandbox error"
            with st.expander(label, expanded="ERROR:" in raw or "[IMAGE]" in raw):
                if "[IMAGE]" in raw:
                    # Split on the [IMAGE]...[/IMAGE] blocks and render each piece
                    # Format: text...[IMAGE]\n<img ... />\n[/IMAGE]text...
                    import re
                    segments = re.split(r"\[IMAGE\](.*?)\[/IMAGE\]", raw, flags=re.DOTALL)
                    for idx, segment in enumerate(segments):
                        if idx % 2 == 0:
                            # Plain text segment
                            if segment.strip():
                                st.code(segment.strip(), language="text")
                        else:
                            # Image segment — contains the raw <img ...> tag
                            img_tag = segment.strip()
                            if img_tag.startswith("<img"):
                                st.markdown(img_tag, unsafe_allow_html=True)
                else:
                    st.code(raw, language="text")
        else:
            with st.expander(f"📤 `{tool_name}`", expanded=False):
                st.text(raw[:3000] + ("…" if len(raw) > 3000 else ""))


# ─────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<span class="status-dot"></span><span style="font-family:\'JetBrains Mono\',monospace;font-size:0.8rem;color:#6b7280">agent online</span>',
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # ── File upload ───────────────────────────────────────────
    st.markdown("### 📎 Attach Files")
    st.caption("Files are sent as context with every message.")

    uploaded_files = st.file_uploader(
        label="Drop files here",
        accept_multiple_files=True,
        type=[
            "csv", "xlsx", "xls",
            "txt", "md", "py", "js", "ts", "json", "yaml", "yml", "toml", "sh",
            "pdf", "docx",
            "png", "jpg", "jpeg", "gif", "webp",
        ],
        label_visibility="collapsed",
    )

    if uploaded_files:
        for f in uploaded_files:
            if f.name not in st.session_state.file_contexts:
                with st.spinner(f"Processing {f.name}…"):
                    content, df, raw = process_file(f)
                st.session_state.file_contexts[f.name] = content
                if raw is not None:
                    st.session_state.files_raw[f.name] = raw
                if df is not None:
                    st.session_state.dataframes[f.name] = df
                st.toast(f"✅ {f.name} loaded", icon="📎")

    # Show loaded files
    if st.session_state.file_contexts:
        st.markdown("**Loaded:**")
        for fname in list(st.session_state.file_contexts.keys()):
            col_name, col_btn = st.columns([5, 1])
            ext = Path(fname).suffix.upper().lstrip(".")
            col_name.markdown(
                f'<span class="file-badge">{ext}</span> `{fname}`',
                unsafe_allow_html=True,
            )
            if col_btn.button("✕", key=f"rm_{fname}", help=f"Remove {fname}"):
                del st.session_state.file_contexts[fname]
                st.session_state.files_raw.pop(fname, None)
                st.session_state.dataframes.pop(fname, None)
                st.rerun()

    st.markdown("---")

    # ── Controls ──────────────────────────────────────────────
    st.markdown("### ⚙️ Controls")

    if st.button("🗑️  Clear conversation", use_container_width='stretch'):
        st.session_state.lc_messages = []
        st.session_state.display_turns = []
        st.rerun()

    if st.button("🗂️  Clear files", use_container_width='stretch'):
        st.session_state.file_contexts = {}
        st.session_state.files_raw = {}
        st.session_state.dataframes = {}
        st.rerun()

    st.markdown("---")
    st.caption(
        "🔗 Azure OpenAI + LangGraph  \n"
        "🐳 Docker sandbox for code execution  \n"
        "🔍 Tavily for web research"
    )


# ─────────────────────────────────────────────────────────────
# Main layout
# ─────────────────────────────────────────────────────────────
st.markdown("# ⚡ Deep Agent Coder")
st.caption("Research · Code · Analyze · Visualize")

# CSV explorers (above chat, collapsible)
if st.session_state.dataframes:
    st.markdown("### 📊 Uploaded Data")
    for fname, df in st.session_state.dataframes.items():
        render_csv_explorer(fname, df)
    st.divider()

# Conversation history
for turn in st.session_state.display_turns:
    render_turn(turn)

# ── Chat input ────────────────────────────────────────────────
if prompt := st.chat_input("Ask anything — research, code, analyze data…"):

    # Show user message immediately (clean version)
    display_prompt = prompt
    if st.session_state.file_contexts:
        fnames = ", ".join(f"`{f}`" for f in st.session_state.file_contexts)
        display_prompt += f"\n\n> *Attached: {fnames}*"

    with st.chat_message("user", avatar="🧑"):
        st.markdown(display_prompt)

    # Build full content (with file context) to send to agent
    full_content = build_user_content(prompt)

    # Run agent
    with st.spinner("⚡ Thinking…"):
        result = asyncio.run(_invoke_agent(full_content))

    if result and "messages" in result:
        all_msgs = result["messages"]

        # Update stored LangChain messages (full history for next turn)
        st.session_state.lc_messages = all_msgs

        # The new messages are everything after what we had before
        # (last element is the final AI message; find the user msg we just sent)
        # Simplest: find the first new AI message onward
        new_msgs = []
        found_our_user_msg = False
        for m in all_msgs:
            mtype = m.__class__.__name__
            if not found_our_user_msg:
                if ("Human" in mtype or mtype == "HumanMessage") and full_content in (
                    m.content if isinstance(m.content, str) else ""
                ):
                    found_our_user_msg = True
                continue
            new_msgs.append(m)

        # Store this turn for display
        turn = {
            "user_display": display_prompt,
            "agent_messages": new_msgs,
        }
        st.session_state.display_turns.append(turn)

        # Render agent messages immediately
        for msg in new_msgs:
            _render_one_message(msg)