"""
Chat with PDF / Excel — Hybrid (chats.create pattern preserved)
────────────────────────────────────────────────────────────────
• Gemini    → file storage + retrieval + chat (uses client.chats.create)
• Claude    → chat only, with context retrieved via Gemini one-off call
• OpenAI    → chat only, with context retrieved via Gemini one-off call

Frontend flow (per question):
    Stage 1 spinner: 🔍 Retrieving context from PDF (Gemini)
    → excerpts expander
    Stage 2 spinner: 💭 {chosen model} is thinking
    → final answer

Multiple file support:
    - Upload many PDFs at once or in batches — each gets its own store
    - Upload many Excel files — all sheets combined with file labels
    - Excerpts retrieved from ALL PDF stores and merged
"""

import streamlit as st
import tempfile
import os
import time
import pandas as pd
from google import genai
from google.genai import types
import anthropic
from openai import OpenAI

# ── Page config ──────────────────────────────────────────────────
st.set_page_config(page_title="Chat with PDF / Excel", page_icon="📄")
st.title("📄 Chat with PDF / Excel")
st.caption("Gemini retrieves · Your chosen model answers")



gemini_key      = st.secrets["GEMINI_API_KEY"]
anthropic_key   = st.secrets["ANTHROPIC_API_KEY"]
openai_key      = st.secrets["OPENAI_API_KEY"]
perplexity_key  = st.secrets["PERPLEXITY_KEY"]
groq_key        = st.secrets["GROQ_KEY"]

if not gemini_key:
    st.error("⚠️ GEMINI_API_KEY required.")
    st.stop()

# ── Clients (once) ───────────────────────────────────────────────
if "gemini_client" not in st.session_state:
    st.session_state.gemini_client = genai.Client(api_key=gemini_key)
if "anthropic_client" not in st.session_state and anthropic_key:
    st.session_state.anthropic_client = anthropic.Anthropic(api_key=anthropic_key)
if "openai_client" not in st.session_state and openai_key:
    st.session_state.openai_client = OpenAI(api_key=openai_key)
if "perplexity_client" not in st.session_state and perplexity_key:
    st.session_state.perplexity_client = OpenAI(
        api_key=perplexity_key,
        base_url="https://api.perplexity.ai",
    )
if "groq_client" not in st.session_state and groq_key:
    st.session_state.groq_client = OpenAI(
        api_key=groq_key,
        base_url="https://api.groq.com/openai/v1",
    )

gemini_client      = st.session_state.gemini_client
anthropic_client   = st.session_state.get("anthropic_client")
openai_client      = st.session_state.get("openai_client")
perplexity_client  = st.session_state.get("perplexity_client")
groq_client        = st.session_state.get("groq_client")

# ── Model registry ───────────────────────────────────────────────
MODELS = {
    # ── Gemini ───────────────────────────────────────────────
    "Gemini 2.5 Flash":           ("gemini", "gemini-2.5-flash"),
    "gemini-3-flash-preview":      ("gemini", "gemini-3-flash-preview"),
    "Gemini 2.5 Pro":             ("gemini", "gemini-2.5-pro"),
    "Gemini 2.5 Flash Lite":      ("gemini", "gemini-2.5-flash-lite"),
    # ── OpenAI ───────────────────────────────────────────────
    "GPT-4o":                     ("openai", "gpt-4o"),
    "GPT-4o-mini":                ("openai", "gpt-4o-mini"),
    "GPT-4.1":               ("openai", "gpt-4.1"),

    # ── Perplexity ───────────────────────────────────────────
    "Perplexity Sonar":           ("perplexity", "sonar"),
    "Perplexity Sonar Pro":       ("perplexity", "sonar-pro"),
    "Perplexity Sonar Reasoning": ("perplexity", "sonar-reasoning"),

    # ── Groq ─────────────────────────────────────────────────
    "Groq Llama 3.3 70B":         ("groq", "llama-3.3-70b-versatile"),
    "Groq Llama 3.1 8B Instant":  ("groq", "llama-3.1-8b-instant"),
    "Groq Mixtral 8x7B":          ("groq", "mixtral-8x7b-32768"),
    "Groq Gemma2 9B":             ("groq", "gemma2-9b-it"),

    # ── Claude ───────────────────────────────────────────────
    "Claude Sonnet 4.6":          ("claude", "claude-sonnet-4-6"),
    "Claude Opus 4.7":            ("claude", "claude-opus-4-7"),
    "Claude Haiku 4.5":           ("claude", "claude-haiku-4-5-20251001"),
}


# ── System prompt builder ─────────────────────────────────────────
def build_system_prompt(excerpts: str | None, excel_context: str | None) -> str:
    parts = []
    if excerpts:
        parts.append(
            "=== PDF EXCERPTS ===\n"
            f"{excerpts}\n"
            "=== END PDF ==="
        )
    if excel_context:
        parts.append(
            "=== EXCEL DATA ===\n"
            f"{excel_context}\n"
            "=== END EXCEL ==="
        )
    context_block = "\n\n".join(parts) if parts else "NO_CONTEXT"

    return (
        "You are a helpful analyst with access to uploaded documents.\n"
        "Use ONLY the content below to answer questions.\n"
        "Always mention which source (PDF filename or Excel sheet) your answer comes from.\n"
        "If the answer is not in the provided content, say so.\n\n"
        + context_block
    )


# ── Helper: Excel → markdown ──────────────────────────────────────
def excel_to_text(excel_bytes: bytes) -> tuple[str, list[str]]:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
        tmp.write(excel_bytes)
        tmp_path = tmp.name
    try:
        xls = pd.ExcelFile(tmp_path)
        parts, sheet_names = [], []
        for name in xls.sheet_names:
            df = xls.parse(name)
            if df.empty:
                continue
            sheet_names.append(name)
            parts.append(f"### Sheet: {name}\n\n{df.to_markdown(index=False)}")
        xls.close()
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
    return "\n\n---\n\n".join(parts), sheet_names


# ── Wait for Gemini store to finish indexing ──────────────────────
def wait_for_store_ready(store_name: str, timeout: int = 180) -> bool:
    """Poll until the file search store state is ACTIVE or timeout."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            store = gemini_client.file_search_stores.get(name=store_name)
            state = str(getattr(store, "state", "ACTIVE")).upper()
            if "ACTIVE" in state:
                return True
            if "ERROR" in state or "FAILED" in state:
                return False
        except Exception:
            pass
        time.sleep(4)
    return False


# ── Stage 1: extract excerpts from one store ──────────────────────
def gemini_extract_excerpts(question: str, store_name: str) -> str:
    prompt = (
        f"User question: {question}\n\n"
        "TASK: Extract 3–6 most relevant VERBATIM excerpts from the PDF that "
        "would help answer this question. Do NOT answer the question yourself.\n\n"
        "Format each excerpt as:\n"
        '  [Section / Page X]: "verbatim text"\n\n'
        "If nothing relevant: NO_RELEVANT_CONTEXT"
    )
    chat = gemini_client.chats.create(
        model="gemini-2.5-flash",
        config=types.GenerateContentConfig(
            tools=[
                types.Tool(
                    file_search=types.FileSearch(
                        file_search_store_names=[store_name]
                    )
                )
            ],
            system_instruction=(
                "You are a helpful assistant. Answer questions about the uploaded PDF. "
                "Cite specific sections when possible."
            ),
        ),
    )
    response = chat.send_message(prompt)
    return (response.text or "").strip()


# ── Extract excerpts from ALL PDF stores, label by filename ───────
def extract_all_excerpts(question: str, store_names: list[str]) -> str:
    all_excerpts = []
    for i, store_name in enumerate(store_names):
        label = st.session_state.pdf_names.get(store_name, f"PDF {i + 1}")
        try:
            excerpts = gemini_extract_excerpts(question, store_name)
        except Exception as e:
            excerpts = f"(Error retrieving from {label}: {e})"
        all_excerpts.append(f"--- From: {label} ---\n{excerpts}")
    return "\n\n".join(all_excerpts)


# ── Stage 2 answer functions ──────────────────────────────────────

def gemini_answer(messages, model_id, system_prompt) -> str:
    # Convert prior turns into Gemini history format (role: user/model)
    history = []
    for m in messages[:-1]:
        role = "model" if m["role"] == "assistant" else "user"
        history.append(
            types.Content(role=role, parts=[types.Part(text=m["content"])])
        )
    chat = gemini_client.chats.create(
        model=model_id,
        config=types.GenerateContentConfig(system_instruction=system_prompt),
        history=history,
    )
    return chat.send_message(messages[-1]["content"]).text


def claude_answer(messages, model_id, system_prompt) -> str:
    if not anthropic_client:
        raise RuntimeError("ANTHROPIC_API_KEY not configured.")
    response = anthropic_client.messages.create(
        model=model_id,
        max_tokens=60000,
        system=system_prompt,
        messages=[{"role": m["role"], "content": m["content"]} for m in messages],
    )
    return next(b.text for b in response.content if b.type == "text")


def openai_answer(messages, model_id, system_prompt) -> str:
    if not openai_client:
        raise RuntimeError("OPENAI_API_KEY not configured.")
    openai_messages = [{"role": "system", "content": system_prompt}]
    openai_messages += [{"role": m["role"], "content": m["content"]} for m in messages]
    response = openai_client.chat.completions.create(
        model=model_id,
        max_tokens=8000,
        messages=openai_messages,
    )
    return response.choices[0].message.content


def perplexity_answer(messages, model_id, system_prompt) -> str:
    if not perplexity_client:
        raise RuntimeError("PERPLEXITY_API_KEY not configured.")
    pplx_messages = [{"role": "system", "content": system_prompt}]
    pplx_messages += [{"role": m["role"], "content": m["content"]} for m in messages]
    response = perplexity_client.chat.completions.create(
        model=model_id,
        max_tokens=8000,
        messages=pplx_messages,
    )
    return response.choices[0].message.content


def groq_answer(messages, model_id, system_prompt) -> str:
    if not groq_client:
        raise RuntimeError("GROQ_API_KEY not configured.")
    groq_messages = [{"role": "system", "content": system_prompt}]
    groq_messages += [{"role": m["role"], "content": m["content"]} for m in messages]
    response = groq_client.chat.completions.create(
        model=model_id,
        max_tokens=8000,
        messages=groq_messages,
    )
    return response.choices[0].message.content


def generate_answer(messages, model_choice, system_prompt) -> str:
    provider, model_id = MODELS[model_choice]
    if provider == "gemini":
        return gemini_answer(messages, model_id, system_prompt)
    if provider == "openai":
        return openai_answer(messages, model_id, system_prompt)
    if provider == "perplexity":
        return perplexity_answer(messages, model_id, system_prompt)
    if provider == "groq":
        return groq_answer(messages, model_id, system_prompt)
    return claude_answer(messages, model_id, system_prompt)


# ── Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Chat Model")
    available = [
        name for name, (provider, _) in MODELS.items()
        if provider == "gemini"
        or (provider == "claude" and anthropic_client)
        or (provider == "openai" and openai_client)
        or (provider == "perplexity" and perplexity_client)
        or (provider == "groq" and groq_client)
    ]
    model_choice = st.selectbox(
        "Pick the answering model", available, index=0,
        help="Gemini always handles PDF retrieval. This model writes the answer.",
    )
    if not anthropic_client:
        st.caption("💡 Add `ANTHROPIC_API_KEY` to unlock Claude.")
    if not openai_client:
        st.caption("💡 Add `OPENAI_API_KEY` to unlock GPT models.")
    if not perplexity_client:
        st.caption("💡 Add `perplexity_key` to unlock Perplexity Sonar.")

    st.divider()
    st.header("📁 Loaded Files")

    store_names = st.session_state.get("store_names", [])
    pdf_names   = st.session_state.get("pdf_names", {})
    for sn in store_names:
        st.success(f"📄 {pdf_names.get(sn, sn)}")

    # Show each Excel file with its sheets
    for ef in st.session_state.get("excel_files", []):
        st.success(f"📊 {ef['name']} — sheets: {', '.join(ef['sheets'])}")

    if store_names or st.session_state.get("excel_files"):
        if st.button("🗑️ Reset All"):
            for sn in store_names:
                try:
                    gemini_client.file_search_stores.delete(name=sn)
                except Exception:
                    pass
            for key in ["store_names", "pdf_names", "messages", "file_keys",
                        "excel_files", "excel_context"]:
                st.session_state.pop(key, None)
            st.rerun()


# ── File uploader — multiple files allowed ────────────────────────
uploaded_files = st.file_uploader(
    "Upload PDF and/or Excel files (multiple allowed)",
    type=["pdf", "xlsx", "xls"],
    accept_multiple_files=True,
)

# ── Process each uploaded file ────────────────────────────────────
if uploaded_files:
    # Initialise session state buckets
    for key, default in [
        ("store_names", []),
        ("pdf_names", {}),
        ("file_keys", set()),
        ("excel_files", []),
        ("excel_context", ""),
        ("messages", []),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    new_pdfs   = [f for f in uploaded_files if not f.name.lower().endswith((".xlsx", ".xls"))
                  and (f.name + str(f.size)) not in st.session_state.file_keys]
    new_excels = [f for f in uploaded_files if f.name.lower().endswith((".xlsx", ".xls"))
                  and (f.name + str(f.size)) not in st.session_state.file_keys]

    # ── Process Excel files ───────────────────────────────────────
    for uploaded in new_excels:
        file_key = uploaded.name + str(uploaded.size)
        try:
            with st.spinner(f"Reading {uploaded.name}…"):
                text_context, sheet_names = excel_to_text(uploaded.read())
            if not sheet_names:
                st.error(f"⚠️ All sheets empty in **{uploaded.name}** — skipped.")
                continue
            # Accumulate: prepend a file-level heading so context stays labelled
            file_block = f"### File: {uploaded.name}\n\n{text_context}"
            separator  = "\n\n---\n\n" if st.session_state.excel_context else ""
            st.session_state.excel_context += separator + file_block
            st.session_state.excel_files.append(
                {"name": uploaded.name, "sheets": sheet_names}
            )
            st.session_state.file_keys.add(file_key)
            st.success(f"✅ Excel loaded: **{uploaded.name}** — sheets: {', '.join(sheet_names)}")
        except Exception as e:
            st.error(f"❌ Failed to read **{uploaded.name}**: {e}")

    # ── Process PDF files ─────────────────────────────────────────
    if new_pdfs:
        progress = st.progress(0, text=f"Processing 0 / {len(new_pdfs)} PDFs…")

    for idx, uploaded in enumerate(new_pdfs):
        file_key = uploaded.name + str(uploaded.size)
        try:
            raw = uploaded.read()
            if len(raw) == 0:
                st.error(f"⚠️ **{uploaded.name}** is empty — skipped.")
                continue

            with st.spinner(f"Creating store for **{uploaded.name}**…"):
                store = gemini_client.file_search_stores.create(
                    config={"display_name": uploaded.name.replace(".pdf", "")}
                )

            with st.spinner(f"Uploading **{uploaded.name}**…"):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(raw)
                    tmp_path = tmp.name
                try:
                    gemini_client.file_search_stores.upload_to_file_search_store(
                        file=tmp_path,
                        file_search_store_name=store.name,
                    )
                finally:
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass

            with st.spinner(f"Indexing **{uploaded.name}** (waiting for Gemini)…"):
                ready = wait_for_store_ready(store.name)

            if not ready:
                st.warning(
                    f"⚠️ **{uploaded.name}** indexing timed out — "
                    "it may still work but results could be incomplete."
                )

            st.session_state.store_names.append(store.name)
            st.session_state.pdf_names[store.name] = uploaded.name
            st.session_state.file_keys.add(file_key)
            st.success(f"✅ PDF indexed: **{uploaded.name}**")

        except Exception as e:
            st.error(f"❌ Failed to process **{uploaded.name}**: {e}")

        finally:
            if new_pdfs:
                progress.progress(
                    (idx + 1) / len(new_pdfs),
                    text=f"Processing {idx + 1} / {len(new_pdfs)} PDFs…",
                )

    if new_pdfs:
        progress.empty()


# ── Guard: need at least one file ─────────────────────────────────
has_pdf   = bool(st.session_state.get("store_names"))
has_excel = bool(st.session_state.get("excel_context"))

if not has_pdf and not has_excel:
    st.info("Upload one or more PDF / Excel files above to start chatting.")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Display history ───────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# ── Chat input ────────────────────────────────────────────────────
if prompt := st.chat_input("Ask anything about your files…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            # Stage 1: retrieve excerpts from ALL PDFs
            excerpts = None
            if has_pdf:
                with st.spinner("🔍 Retrieving context from PDF(s) (Gemini)…"):
                    excerpts = extract_all_excerpts(
                        prompt, st.session_state.store_names
                    )
                with st.expander("📚 Retrieved excerpts", expanded=False):
                    st.markdown(excerpts or "_(no excerpts)_")

            # Stage 2: build unified prompt + answer
            excel_context = st.session_state.get("excel_context") or None
            system_prompt = build_system_prompt(excerpts, excel_context)

            with st.spinner(f"💭 {model_choice} is thinking…"):
                reply = generate_answer(
                    st.session_state.messages,
                    model_choice,
                    system_prompt,
                )

            st.markdown(reply)
            st.caption(f"_Answered by **{model_choice}**_")

        except Exception as e:
            reply = f"❌ Error: {e}"
            st.error(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})
