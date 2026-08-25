"""Điểm khởi chạy giao diện Gradio của learnbot_ai.

Module này định nghĩa bố cục, liên kết sự kiện, điều phối xử lý tài liệu và
hiển thị số liệu hệ thống. Logic RAG chính nằm trong ``core`` và ``features``.
"""

import os
import logging
import webbrowser
import gradio as gr
from typing import List, Tuple, Optional
from datetime import datetime

# Cấu hình
from config import (
    DEFAULT_MODEL_CHOICE, GEMINI_MODEL_NAME, OPENAI_MODEL_NAME,
    SILICONFLOW_MODEL_NAME,
    MODEL_CHOICES, MODEL_DISPLAY_NAMES, is_configured_api_key
)

# Các module RAG chính
from core.embeddings import EMBED_MODEL_NAME
from core.vector_store import vector_store
from core.bm25_index import tokenize_vietnamese
from core.ingestion import DocumentSource, ingest_documents
from core.generator import query_answer
from llm_provider import call_llm, get_provider_config, get_provider_name

# Tiện ích
from utils.network import is_port_available

logging.basicConfig(level=logging.INFO)
print("Phiên bản Gradio:", gr.__version__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Xử lý tài liệu
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def process_multiple_files(files, progress=gr.Progress()):
    """Cầu nối tương thích giữa thành phần tải tệp Gradio và pipeline lõi."""
    sources = [
        DocumentSource(path=file.name, display_name=os.path.basename(file.name))
        for file in (files or [])
    ]
    try:
        result = ingest_documents(sources, progress=progress)
        return result.message, result.filenames
    except Exception as exc:
        logging.error("Quá trình xử lý tài liệu gặp lỗi: %s", exc)
        return f"Không thể xử lý tài liệu: {exc}", []


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Hiển thị phân đoạn
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
chunk_data_cache = {}


def get_document_chunks(progress=gr.Progress()):
    """Chuẩn bị dữ liệu phân đoạn để hiển thị trên giao diện."""
    global chunk_data_cache
    try:
        progress(0.1, desc="Đang tải dữ liệu...")
        chunk_data_cache.clear()

        if not vector_store.id_order:
            return [], "Kho tri thức chưa có dữ liệu. Hãy tải và xử lý tài liệu trước."

        table_data = []
        for idx, chunk_id in enumerate(vector_store.id_order):
            content = vector_store.contents_map.get(chunk_id, "")
            meta = vector_store.metadatas_map.get(chunk_id, {})
            if not content:
                continue
            chunk_data = {
                "row_id": idx, "chunk_id": chunk_id,
                "source": meta.get("source", "Không rõ nguồn"),
                "page": meta.get("page"),
                "content": content,
                "preview": content[:200] + "..." if len(content) > 200 else content,
                "char_count": len(content),
                "token_count": len(tokenize_vietnamese(content))
            }
            chunk_data_cache[idx] = chunk_data
            table_data.append([
                chunk_data["source"],
                chunk_data["page"] if chunk_data["page"] is not None else "—",
                f"{idx + 1}/{len(vector_store.id_order)}",
                chunk_data["char_count"], chunk_data["token_count"], chunk_data["preview"]
            ])

        progress(1.0, desc="Hoàn tất")
        return table_data, f"Có {len(table_data)} phân đoạn."
    except Exception as e:
        chunk_data_cache.clear()
        return [], f"Không thể tải dữ liệu phân đoạn: {e}"


def show_chunk_details(evt: gr.SelectData):
    """Hiển thị nội dung đầy đủ của phân đoạn được chọn."""
    try:
        if not evt.index or evt.index[0] is None:
            return "Vui lòng chọn một hàng hợp lệ."
        selected = chunk_data_cache.get(evt.index[0])
        if not selected:
            return "Không tìm thấy dữ liệu của phân đoạn này."
        page = selected["page"] if selected["page"] is not None else "Không áp dụng"
        return f"""[Nguồn] {selected['source']}
[Trang] {page}
[Mã phân đoạn] {selected['chunk_id']}
[Số ký tự] {selected['char_count']}
[Số từ] {selected['token_count']}
----------------------------
{selected['content']}"""
    except Exception as e:
        return f"Không thể hiển thị phân đoạn: {e}"


def get_system_models_info():
    """Trả về thông tin mô hình và kỹ thuật đang sử dụng."""
    return {
        "Mô hình embedding": EMBED_MODEL_NAME,
        "Cách phân đoạn": "RecursiveCharacterTextSplitter (400 ký tự, chồng lấn 40)",
        "Phương pháp truy xuất": "Tìm kiếm vector + BM25 kết hợp (α=0,7)",
        "Mô hình xếp hạng lại": "CrossEncoder đa ngôn ngữ",
        "Dịch vụ LLM mặc định": get_model_display_name(DEFAULT_MODEL_CHOICE),
        "Mô hình SiliconFlow": SILICONFLOW_MODEL_NAME,
        "Mô hình OpenAI": OPENAI_MODEL_NAME,
        "Mô hình Gemini": GEMINI_MODEL_NAME,
        "Công cụ tách từ": "Underthesea (tiếng Việt)"
    }


def get_model_display_name(model_choice_val):
    """Tên provider hiển thị trên giao diện."""
    return MODEL_DISPLAY_NAMES.get(
        model_choice_val,
        f"Dịch vụ không xác định ({model_choice_val})",
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Giao diện Gradio 6.x
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
APP_THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.amber,
    secondary_hue=gr.themes.colors.slate,
    neutral_hue=gr.themes.colors.stone,
    font=[
        gr.themes.Font("Segoe UI Variable Text"),
        "Segoe UI",
        "sans-serif",
    ],
    font_mono=["Cascadia Code", "Consolas", "monospace"],
    radius_size=gr.themes.sizes.radius_lg,
    spacing_size=gr.themes.sizes.spacing_md,
).set(
    body_background_fill="#faf9f5",
    body_background_fill_dark="#141413",
    body_text_color="#141413",
    body_text_color_dark="#faf9f5",
    body_text_color_subdued="#68665f",
    body_text_color_subdued_dark="#b0aea5",
    background_fill_primary="#fffefa",
    background_fill_primary_dark="#1b1b19",
    background_fill_secondary="#f3f1e9",
    background_fill_secondary_dark="#22221f",
    block_background_fill="#fffefa",
    block_background_fill_dark="#1b1b19",
    block_border_color="rgba(20,20,19,0.08)",
    block_border_color_dark="rgba(250,249,245,0.10)",
    block_radius="24px",
    block_shadow="none",
    input_background_fill="#fffefa",
    input_background_fill_dark="#181816",
    input_border_color="rgba(20,20,19,0.12)",
    input_border_color_dark="rgba(250,249,245,0.14)",
    input_border_color_focus="#d97757",
    input_border_color_focus_dark="#d97757",
    input_radius="18px",
    button_primary_background_fill="#d97757",
    button_primary_background_fill_dark="#d97757",
    button_primary_background_fill_hover="#c76647",
    button_primary_background_fill_hover_dark="#e18769",
    button_primary_text_color="#141413",
    button_primary_text_color_dark="#141413",
    button_primary_border_color="#d97757",
    button_primary_border_color_dark="#d97757",
    button_primary_shadow="none",
    button_primary_shadow_hover="0 14px 34px rgba(168, 75, 47, 0.18)",
    button_secondary_background_fill="#f1efe7",
    button_secondary_background_fill_dark="#242421",
    button_secondary_border_color="rgba(20,20,19,0.08)",
    button_secondary_border_color_dark="rgba(250,249,245,0.10)",
    button_secondary_text_color="#141413",
    button_secondary_text_color_dark="#faf9f5",
    button_transform_hover="none",
    button_transition="transform 700ms cubic-bezier(0.32,0.72,0,1), background-color 700ms cubic-bezier(0.32,0.72,0,1), box-shadow 700ms cubic-bezier(0.32,0.72,0,1)",
)

CSS = """
:root {
    --app-bg:#faf9f5;
    --app-surface:#fffefa;
    --app-surface-muted:#f1efe7;
    --app-shell:#ebe8de;
    --app-text:#141413;
    --app-muted:#68665f;
    --app-mid:#b0aea5;
    --app-accent:#d97757;
    --app-accent-strong:#bd5d40;
    --app-accent-soft:#f7e7df;
    --app-info:#6a9bcc;
    --app-success:#788c5d;
    --app-danger:#b95446;
    --hairline:rgba(20,20,19,.09);
    --app-shadow:0 30px 80px rgba(76,62,47,.10),0 8px 24px rgba(76,62,47,.05);
    --app-shadow-raised:0 22px 55px rgba(139,79,55,.16);
    --motion:cubic-bezier(.32,.72,0,1);
    --font-display:"Segoe UI Variable Display","Segoe UI",sans-serif;
    --font-body:"Segoe UI Variable Text","Segoe UI",sans-serif;
}
body.dark {
    --app-bg:#141413;
    --app-surface:#1b1b19;
    --app-surface-muted:#22221f;
    --app-shell:#292824;
    --app-text:#faf9f5;
    --app-muted:#b0aea5;
    --app-mid:#79776f;
    --app-accent:#d97757;
    --app-accent-strong:#e18769;
    --app-accent-soft:#3a241d;
    --app-info:#83acd3;
    --app-success:#91a876;
    --app-danger:#df806f;
    --hairline:rgba(250,249,245,.11);
    --app-shadow:0 36px 90px rgba(0,0,0,.30),0 10px 30px rgba(0,0,0,.18);
    --app-shadow-raised:0 24px 60px rgba(0,0,0,.34);
}
html { scroll-behavior:smooth; }
body {
    background:
        radial-gradient(circle at 8% -10%, color-mix(in srgb,var(--app-accent) 12%,transparent), transparent 29rem),
        radial-gradient(circle at 92% 2%, color-mix(in srgb,var(--app-info) 8%,transparent), transparent 32rem),
        var(--app-bg)!important;
    color:var(--app-text)!important;
    font-family:var(--font-body)!important;
}
body::before {
    content:""; position:fixed; inset:0; z-index:10; pointer-events:none; opacity:.032;
    background-image:url("data:image/svg+xml,%3Csvg viewBox='0 0 180 180' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='.55'/%3E%3C/svg%3E");
}
.gradio-container {
    max-width:1500px!important;
    width:100%!important;
    margin:0 auto!important;
    padding:clamp(18px,3vw,42px) clamp(16px,4vw,64px) 72px!important;
    overflow-x:hidden;
}
.gradio-container > .main { width:100%!important; padding:0!important; }
footer { display:none!important; }
.skip-link {
    position:fixed; top:12px; left:12px; z-index:30; padding:10px 16px; border-radius:999px;
    background:var(--app-text); color:var(--app-bg); font-family:var(--font-display);
    opacity:0; pointer-events:none; transform:translateY(-150%);
    transition:opacity 180ms ease,transform 500ms var(--motion);
}
.skip-link:focus-visible { opacity:1; pointer-events:auto; transform:translateY(0); }
h1,h2,h3,h4,h5,h6,strong,label,button,[role="tab"] { font-family:var(--font-display)!important; }
p { text-wrap:pretty; }
button, [role="button"], [role="tab"] {
    cursor:pointer!important; transition:transform 700ms var(--motion),background-color 700ms var(--motion),color 700ms var(--motion),box-shadow 700ms var(--motion)!important;
}
button:hover, [role="button"]:hover { transform:translateY(-2px); }
button:active, [role="button"]:active { transform:scale(.98); }
button:focus-visible, [role="button"]:focus-visible, [role="tab"]:focus-visible,
textarea:focus-visible, input:focus-visible {
    outline:3px solid color-mix(in srgb,var(--app-accent) 52%,transparent)!important;
    outline-offset:3px!important;
}
.topbar {
    width:100%!important; align-items:center!important; margin:0 auto 26px!important; padding:10px 10px 10px 12px!important;
    border-radius:30px!important; background:color-mix(in srgb,var(--app-surface) 88%,transparent)!important;
    box-shadow:inset 0 0 0 1px var(--hairline),0 18px 55px rgba(76,62,47,.08)!important;
}
.brand-shell { display:flex; align-items:center; gap:16px; padding:2px 0; }
.brand-mark {
    display:grid; place-items:center; flex:0 0 54px; width:54px; height:54px;
    border-radius:18px; color:#faf9f5; background:var(--app-accent);
    box-shadow:inset 0 1px 1px rgba(255,255,255,.30),0 16px 30px rgba(168,75,47,.20);
}
.brand-mark svg { width:26px; height:26px; }
.brand-eyebrow {
    margin:0 0 5px; color:var(--app-accent-strong); font-family:var(--font-display)!important;
    font-size:10px; font-weight:600; letter-spacing:.2em; text-transform:uppercase;
}
.brand-title {
    margin:0; max-width:780px; color:var(--app-text); font-size:clamp(25px,3.2vw,46px);
    line-height:1.02; letter-spacing:-.048em; font-weight:600; text-wrap:balance;
}
.brand-copy { margin:11px 0 0; max-width:66ch; color:var(--app-muted); font-size:14px; line-height:1.72; }
.trust-row { display:flex; flex-wrap:wrap; gap:9px; margin-top:15px; }
.trust-chip {
    display:inline-flex; align-items:center; min-height:28px; padding:4px 11px; border-radius:999px;
    color:var(--app-muted); background:var(--app-surface-muted); box-shadow:inset 0 0 0 1px var(--hairline);
    font-family:var(--font-display); font-size:10px; font-weight:500; letter-spacing:.02em;
}
.theme-toggle-btn {
    min-width:132px!important; min-height:48px!important; padding:8px 12px 8px 17px!important;
    border:0!important; border-radius:999px!important; font-size:11px!important; font-weight:600!important;
    letter-spacing:.03em!important; background:var(--app-surface-muted)!important;
    box-shadow:inset 0 0 0 1px var(--hairline)!important;
}
.theme-toggle-btn::after {
    content:"◐"; display:grid; place-items:center; width:30px; height:30px; margin-left:8px;
    border-radius:50%; color:#faf9f5; background:var(--app-text); transition:transform 700ms var(--motion);
}
.theme-toggle-btn:hover::after { transform:rotate(180deg) scale(1.05); }
.main-tabs > .tab-nav, .main-tabs [role="tablist"] {
    width:max-content!important; max-width:100%!important; gap:5px!important; margin:0 auto 8px!important;
    padding:6px!important; border:0!important; border-radius:999px!important;
    background:var(--app-shell)!important; box-shadow:inset 0 0 0 1px var(--hairline)!important;
}
.main-tabs { width:100%!important; }
.main-tabs [role="tab"] {
    min-height:42px!important; padding:9px 18px!important; border:0!important; border-radius:999px!important;
    color:var(--app-muted)!important; font-size:11px!important; font-weight:600!important; letter-spacing:.025em!important;
}
.main-tabs [role="tab"][aria-selected="true"] {
    color:var(--app-text)!important; background:var(--app-surface)!important;
    box-shadow:0 8px 22px rgba(76,62,47,.10),inset 0 0 0 1px var(--hairline)!important;
}
.workspace-grid { align-items:flex-start!important; gap:clamp(18px,2vw,30px)!important; padding-top:28px!important; }
.bezel-shell {
    padding:1px!important; border:0!important; border-radius:29px!important; background:var(--app-shell)!important;
    box-shadow:inset 0 0 0 1px var(--hairline),0 22px 60px rgba(76,62,47,.09)!important;
}
.bezel-core {
    padding:clamp(18px,2.2vw,28px)!important; border:0!important; border-radius:28px!important;
    background:var(--app-surface)!important; box-shadow:inset 0 1px 1px rgba(255,255,255,.34)!important;
}
.bezel-core > .bezel-core { padding:0!important; background:transparent!important; box-shadow:none!important; }
.document-panel { position:sticky; top:20px; }
.conversation-panel { margin-top:0!important; }
.section-heading { display:flex; align-items:flex-start; gap:13px; margin:0 0 21px; }
.section-number {
    display:grid; place-items:center; flex:0 0 34px; width:34px; height:26px; border-radius:999px;
    background:var(--app-accent-soft); color:var(--app-accent-strong);
    font-family:var(--font-display); font-size:9px; font-weight:600; letter-spacing:.12em;
}
.section-heading h2 { margin:0; color:var(--app-text); font-size:clamp(18px,1.6vw,23px); line-height:1.18; letter-spacing:-.025em; font-weight:600; }
.section-heading p { margin:6px 0 0; max-width:48ch; color:var(--app-muted); font-size:13px; line-height:1.65; }
.format-note {
    margin:0 0 12px!important; padding:0!important; overflow:visible!important;
    color:var(--app-muted)!important; font-family:var(--font-display)!important; font-size:9px!important;
    line-height:1.5!important; letter-spacing:.1em!important; text-transform:uppercase;
}
#upload-zone {
    min-height:184px!important; border:0!important; border-radius:22px!important; background:var(--app-surface-muted)!important;
    box-shadow:inset 0 0 0 1px var(--hairline),inset 0 1px 1px rgba(255,255,255,.28)!important;
    transition:transform 700ms var(--motion),background-color 700ms var(--motion),box-shadow 700ms var(--motion)!important;
}
#upload-zone:hover { transform:translateY(-3px); background:var(--app-accent-soft)!important; box-shadow:inset 0 0 0 1px color-mix(in srgb,var(--app-accent) 38%,transparent)!important; }
.upload-action,.send-button {
    min-height:50px!important; margin-top:12px!important; border:0!important; border-radius:999px!important;
    color:#141413!important; box-shadow:var(--app-shadow-raised)!important; font-size:12px!important; letter-spacing:.02em!important;
}
.upload-action::after,.send-button::after {
    content:"↗"; display:grid; place-items:center; width:31px; height:31px; margin-left:9px; border-radius:50%;
    background:rgba(250,249,245,.18); transition:transform 700ms var(--motion);
}
.upload-action:hover::after,.send-button:hover::after { transform:translate(3px,-1px) scale(1.06); }
.process-details { margin-top:14px!important; border:0!important; border-radius:18px!important; background:var(--app-surface-muted)!important; box-shadow:inset 0 0 0 1px var(--hairline)!important; }
.file-list { margin-top:8px!important; }
.chat-container {
    min-height:520px!important; border:0!important; border-radius:22px!important; background:var(--app-surface-muted)!important;
    box-shadow:inset 0 0 0 1px var(--hairline)!important;
}
.chat-container .message {
    border:0!important; border-radius:20px!important; box-shadow:inset 0 0 0 1px var(--hairline),0 10px 28px rgba(76,62,47,.06)!important;
}
.chat-container .message.user { background:var(--app-accent-soft)!important; }
.chat-container .message.bot { background:var(--app-surface)!important; }
.chat-container .message p { line-height:1.72!important; }
.composer-card {
    margin-top:15px!important; padding:1px!important; border:0!important; border-radius:25px!important;
    background:var(--app-shell)!important; box-shadow:inset 0 0 0 1px var(--hairline),0 18px 42px rgba(76,62,47,.08)!important;
}
.composer-card > .composer-card {
    padding:14px!important; border-radius:24px!important; background:var(--app-surface)!important;
    box-shadow:inset 0 1px 1px rgba(255,255,255,.3)!important;
}
.composer-card textarea { line-height:1.55!important; }
.composer-options { align-items:end!important; gap:12px!important; }
.composer-actions { align-items:center!important; gap:10px!important; }
.clear-button { min-height:46px!important; border:0!important; border-radius:999px!important; }
.api-info {
    display:flex; flex-wrap:wrap; gap:8px; margin:12px 2px 0; color:var(--app-muted); font-size:10px;
}
.api-badge {
    display:inline-flex; align-items:center; min-height:27px; padding:4px 10px; border-radius:999px;
    background:var(--app-surface-muted); box-shadow:inset 0 0 0 1px var(--hairline);
}
.api-badge::before { content:""; width:6px; height:6px; margin-right:7px; border-radius:50%; background:var(--app-success); }
.api-badge strong { margin-left:4px; color:var(--app-text); }
.footer-note {
    display:block; margin:12px 0 0; padding:0 8px 4px; overflow:visible;
    color:var(--app-muted); font-size:10px; line-height:1.65;
}
.chunk-layout { align-items:flex-start!important; padding-top:28px!important; gap:26px!important; }
.model-card,.monitor-core {
    padding:18px!important; border:0!important; border-radius:20px!important; background:var(--app-surface-muted)!important;
    box-shadow:inset 0 0 0 1px var(--hairline)!important;
}
.model-card > .model-card,.monitor-core > .monitor-core { padding:0!important; background:transparent!important; box-shadow:none!important; }
.chunk-table { border-radius:16px!important; overflow:hidden!important; }
.chunk-detail-box { min-height:200px; font-family:"Cascadia Code",Consolas,monospace; white-space:pre-wrap; }
.system-stack { gap:24px!important; padding-top:28px!important; }
.metrics-grid { gap:14px!important; }
.metric-card {
    min-width:180px!important; padding:20px!important; border:0!important; border-radius:22px!important;
    background:var(--app-surface-muted)!important; box-shadow:inset 0 0 0 1px var(--hairline)!important;
    transition:transform 800ms var(--motion),background-color 800ms var(--motion)!important;
}
.metric-card:nth-child(2) { transform:translateY(18px); }
.metric-card:nth-child(4) { transform:translateY(10px); }
.metric-card:hover { transform:translateY(-4px); }
.metric-title { margin-bottom:12px!important; color:var(--app-muted)!important; font-size:10px!important; letter-spacing:.06em!important; }
.metric-value { margin-bottom:3px!important; color:var(--app-text)!important; font-size:clamp(23px,2.4vw,34px)!important; font-weight:600!important; font-variant-numeric:tabular-nums; }
.metric-trend { color:var(--app-success)!important; font-size:11px!important; }
.progress-container { width:100%; height:6px; margin:13px 0; overflow:hidden; border-radius:999px; background:var(--app-shell); }
.progress-bar { width:100%; height:6px; transform:scaleX(0); transform-origin:left center; border-radius:999px; background:var(--app-accent); transition:transform 900ms var(--motion); }
.log-container {
    min-height:116px; max-height:300px; overflow-y:auto; padding:16px; border:0; border-radius:18px;
    background:var(--app-surface-muted); box-shadow:inset 0 0 0 1px var(--hairline);
    font-family:"Cascadia Code",Consolas,monospace; font-size:11px;
}
.reveal { opacity:1; transform:none; }
body.ui-ready .reveal { opacity:0; transform:translateY(38px); filter:blur(5px); }
body.ui-ready .reveal.is-visible {
    opacity:1; transform:translateY(0); filter:blur(0);
    transition:opacity 900ms var(--motion),transform 900ms var(--motion),filter 900ms var(--motion);
    transition-delay:var(--reveal-delay,0ms);
}
@media (max-width: 900px) {
    .gradio-container { padding:16px 14px 40px!important; }
    .topbar { gap:10px!important; }
    .workspace-grid,.chunk-layout { flex-direction:column!important; }
    .workspace-grid > *, .chunk-layout > * { width:100%!important; min-width:0!important; }
    .document-panel,.conversation-panel { position:static; margin-top:0!important; transform:none; }
    .bezel-shell { border-radius:25px!important; }
    .bezel-core { padding:20px!important; border-radius:24px!important; }
    .chat-container { min-height:430px!important; height:430px!important; }
    .metrics-grid { flex-wrap:wrap!important; }
    .metric-card { flex:1 1 44%!important; }
    .metric-card:nth-child(2),.metric-card:nth-child(4) { transform:none; }
}
@media (max-width: 560px) {
    .gradio-container { padding:12px 10px 36px!important; }
    .topbar,.main-tabs { width:100%!important; min-width:0!important; }
    .topbar { margin-bottom:16px!important; padding:16px!important; border-radius:24px!important; }
    .brand-shell { gap:11px; }
    .brand-mark { flex-basis:44px; width:44px; height:44px; border-radius:14px; }
    .brand-title { font-size:25px; line-height:1.08; }
    .brand-copy { font-size:12px; }
    .trust-row { display:none; }
    .theme-toggle-btn { width:100%!important; min-width:0!important; }
    .main-tabs > .tab-nav,.main-tabs [role="tablist"] { width:100%!important; justify-content:center!important; }
    .main-tabs [role="tab"] { padding:8px 11px!important; font-size:10px!important; }
    .composer-options, .composer-actions { flex-direction:column!important; align-items:stretch!important; }
    .composer-options > *, .composer-actions > * {
        width:100%!important; min-width:0!important; flex:1 1 auto!important;
    }
    .metric-card { flex-basis:100%!important; }
    .chat-container { min-height:360px!important; height:360px!important; }
}
@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after { scroll-behavior:auto!important; transition-duration:0.01ms!important; animation-duration:0.01ms!important; }
    body.ui-ready .reveal { opacity:1; transform:none; filter:none; }
}
"""

# Chuyển chế độ sáng/tối bằng class ``dark`` trên phần tử body.
THEME_JS = """
(() => {
    const applySavedTheme = () => {
        const saved = localStorage.getItem('rag-theme');
        if (saved === 'dark') document.body.classList.add('dark');
    };
    const translations = new Map([
        ['Drop File Here', 'Kéo thả tài liệu vào đây'],
        ['Drop files here', 'Kéo thả tài liệu vào đây'],
        ['Click to Upload', 'Chọn tài liệu'],
        ['- or -', '— hoặc —'],
        ['or', 'hoặc']
    ]);
    const translateUpload = () => {
        const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
        let node;
        while ((node = walker.nextNode())) {
            const value = node.nodeValue?.trim();
            if (translations.has(value)) node.nodeValue = node.nodeValue.replace(value, translations.get(value));
        }
    };
    const observed = new WeakSet();
    const revealObserver = new IntersectionObserver((entries) => {
        entries.forEach((entry) => {
            if (!entry.isIntersecting) return;
            entry.target.classList.add('is-visible');
            revealObserver.unobserve(entry.target);
        });
    }, { threshold: 0.08, rootMargin: '0px 0px -24px 0px' });
    const registerReveals = () => {
        document.querySelectorAll('.reveal').forEach((element, index) => {
            if (observed.has(element)) return;
            observed.add(element);
            element.style.setProperty('--reveal-delay', `${Math.min(index * 80, 240)}ms`);
            revealObserver.observe(element);
        });
    };
    applySavedTheme();
    document.body.classList.add('ui-ready');
    window.setTimeout(() => {
        translateUpload();
        registerReveals();
    }, 120);
    const observer = new MutationObserver(() => {
        translateUpload();
        registerReveals();
    });
    observer.observe(document.body, { childList:true, subtree:true });
})()
"""

def toggle_theme():
    """Để JavaScript của sự kiện nút chuyển chế độ giao diện thực thi."""
    return gr.update()


def build_api_info_html(enable_web_search, model_choice_val):
    """Tạo dải trạng thái ngắn cho cấu hình câu hỏi hiện tại."""
    web_status = "Đã bật" if enable_web_search else "Đã tắt"
    return f"""<div class="api-info" aria-label="Thiết lập câu hỏi">
        <span class="api-badge">Tìm kiếm web <strong>{web_status}</strong></span>
        <span class="api-badge">Dịch vụ LLM <strong>{get_model_display_name(model_choice_val)}</strong></span>
    </div>"""


with gr.Blocks(
    title="learnbot_ai – Trợ lý hỏi đáp tài liệu",
    fill_width=True,
) as demo:
    gr.HTML('<a class="skip-link" href="#noi-dung-chinh">Chuyển đến nội dung chính</a>')
    with gr.Row(elem_classes=["topbar", "reveal"]):
        with gr.Column(scale=10, min_width=300):
            gr.HTML("""
                <div class="brand-shell">
                    <div class="brand-mark" aria-hidden="true">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
                            <path d="M2 4.5A2.5 2.5 0 0 1 4.5 2H12v19H4.5A2.5 2.5 0 0 1 2 18.5z"/>
                            <path d="M22 4.5A2.5 2.5 0 0 0 19.5 2H12v19h7.5a2.5 2.5 0 0 0 2.5-2.5z"/>
                            <path d="M6 7h2.5M15.5 7H18M6 11h3M15 11h3"/>
                        </svg>
                    </div>
                    <div>
                        <p class="brand-eyebrow">learnbot_ai · không gian tri thức</p>
                        <h1 class="brand-title">Đọc sâu hơn. Trả lời có căn cứ.</h1>
                        <p class="brand-copy">Biến tài liệu tiếng Việt thành một kho tri thức có thể đối thoại — mỗi câu trả lời đều đi kèm nguồn và số trang để bạn kiểm chứng.</p>
                        <div class="trust-row" aria-label="Năng lực chính">
                            <span class="trust-chip">Nguồn rõ theo trang</span>
                            <span class="trust-chip">Truy xuất lai FAISS + BM25</span>
                            <span class="trust-chip">LLM bảo mật qua API</span>
                        </div>
                    </div>
                </div>
            """)
        with gr.Column(scale=2, min_width=130):
            theme_btn = gr.Button(
                "Đổi giao diện",
                min_width=112,
                elem_classes="theme-toggle-btn",
            )

    with gr.Tabs(elem_id="noi-dung-chinh", elem_classes="main-tabs") as tabs:
        # Thẻ hỏi đáp
        with gr.TabItem("Tra cứu"):
            with gr.Row(equal_height=False, elem_classes="workspace-grid"):
                with gr.Column(
                    scale=4,
                    min_width=300,
                    elem_classes=["bezel-shell", "document-panel", "reveal"],
                ):
                    with gr.Group(elem_classes="bezel-core"):
                        gr.HTML("""<div class="section-heading">
                            <span class="section-number">01</span>
                            <div><h2>Mở kho tài liệu</h2><p>Thêm tài liệu rồi lập chỉ mục để bắt đầu tra cứu.</p></div>
                        </div>""")
                        gr.HTML(
                            "<p class=\"format-note\">PDF · Word · Excel · PowerPoint · TXT · Markdown</p>"
                        )
                        file_input = gr.File(
                            label="Tài liệu nguồn",
                            file_types=[".pdf", ".txt", ".docx", ".xlsx", ".xls", ".pptx", ".md"],
                            file_count="multiple",
                            height=184,
                            elem_id="upload-zone",
                        )
                        upload_btn = gr.Button(
                            "Lập chỉ mục tài liệu",
                            variant="primary",
                            elem_classes="upload-action",
                        )
                        with gr.Accordion(
                            "Kết quả xử lý",
                            open=True,
                            elem_classes="process-details",
                        ):
                            upload_status = gr.Textbox(
                                label="Trạng thái xử lý",
                                interactive=False,
                                lines=2,
                            )
                            file_list = gr.Textbox(
                                label="Tài liệu đã xử lý",
                                interactive=False,
                                lines=1,
                                elem_classes="file-list",
                            )

                with gr.Column(
                    scale=7,
                    min_width=420,
                    elem_classes=["bezel-shell", "conversation-panel", "reveal"],
                ):
                    with gr.Group(elem_classes="bezel-core"):
                        gr.HTML("""<div class="section-heading">
                            <span class="section-number">02</span>
                            <div><h2>Đối thoại với nguồn</h2><p>Đặt câu hỏi tự nhiên; hệ thống sẽ trả lời từ nội dung đã truy xuất.</p></div>
                        </div>""")
                        chatbot = gr.Chatbot(
                            label="Lịch sử trò chuyện",
                            height=520,
                            elem_classes="chat-container",
                            show_label=False,
                            layout="bubble",
                            buttons=["copy"],
                            placeholder="Kho tri thức đang chờ câu hỏi đầu tiên của bạn.",
                        )
                        api_info = gr.HTML(
                            build_api_info_html(False, DEFAULT_MODEL_CHOICE)
                        )
                        with gr.Group(elem_classes="composer-card"):
                            question_input = gr.Textbox(
                                label="Câu hỏi về tài liệu",
                                lines=2,
                                max_lines=6,
                                placeholder="Ví dụ: So sánh hai luận điểm chính và dẫn nguồn theo từng trang...",
                                autofocus=False,
                            )
                            with gr.Row(elem_classes="composer-options"):
                                web_search_checkbox = gr.Checkbox(
                                    label="Tìm thêm trên web",
                                    value=False,
                                    info="Cần SERPAPI_KEY",
                                )
                                model_choice = gr.Dropdown(
                                    choices=MODEL_CHOICES,
                                    value=DEFAULT_MODEL_CHOICE,
                                    label="Dịch vụ LLM",
                                    info="Mô hình gọi qua API",
                                )
                            with gr.Row(elem_classes="composer-actions"):
                                ask_btn = gr.Button(
                                    "Gửi câu hỏi",
                                    variant="primary",
                                    scale=2,
                                    elem_classes="send-button",
                                )
                                clear_btn = gr.Button(
                                    "Xóa hội thoại",
                                    variant="secondary",
                                    elem_classes="clear-button",
                                    scale=1,
                                )
                        status_display = gr.HTML("")
                        gr.Markdown("""<div class="footer-note">
                            Enter để gửi · Shift + Enter để xuống dòng · Luôn kiểm tra trích dẫn trước khi dùng thông tin quan trọng.
                        </div>""")

        # Thẻ xem phân đoạn
        with gr.TabItem("Kho phân đoạn"):
            with gr.Row(elem_classes="chunk-layout"):
                with gr.Column(
                    scale=1,
                    min_width=280,
                    elem_classes=["bezel-shell", "reveal"],
                ):
                    with gr.Group(elem_classes="bezel-core"):
                        gr.HTML("""<div class="section-heading">
                            <span class="section-number">A</span>
                            <div><h2>Cấu hình truy xuất</h2><p>Mô hình và kỹ thuật đang vận hành phía sau câu trả lời.</p></div>
                        </div>""")
                        models_info = get_system_models_info()
                        with gr.Group(elem_classes="model-card"):
                            for key, value in models_info.items():
                                with gr.Row():
                                    gr.Markdown(f"**{key}**:")
                                    gr.Markdown(f"{value}")
                with gr.Column(
                    scale=2,
                    min_width=460,
                    elem_classes=["bezel-shell", "reveal"],
                ):
                    with gr.Group(elem_classes="bezel-core"):
                        gr.HTML("""<div class="section-heading">
                            <span class="section-number">B</span>
                            <div><h2>Dữ liệu đã lập chỉ mục</h2><p>Kiểm tra từng đoạn nội dung trước khi hệ thống sử dụng.</p></div>
                        </div>""")
                        refresh_chunks_btn = gr.Button(
                            "Làm mới danh sách",
                            variant="primary",
                        )
                        chunks_status = gr.Markdown("Chọn nút trên để xem các phân đoạn đã lập chỉ mục.")
                        chunks_data = gr.Dataframe(
                            headers=["Nguồn", "Trang", "Thứ tự", "Số ký tự", "Số từ", "Nội dung xem trước"],
                            elem_classes="chunk-table",
                            interactive=False,
                            wrap=True,
                            row_count=10,
                        )
                        chunk_detail_text = gr.Textbox(
                            label="Nội dung đầy đủ",
                            placeholder="Chọn một hàng trong bảng để xem toàn bộ nội dung.",
                            lines=8,
                            elem_classes="chunk-detail-box",
                        )

        # Thẻ giám sát hệ thống
        with gr.TabItem("Vận hành"):
            with gr.Column(elem_classes="system-stack"):
                with gr.Group(elem_classes=["bezel-shell", "reveal"]):
                    with gr.Group(elem_classes="bezel-core"):
                        gr.HTML("""<div class="section-heading">
                            <span class="section-number">01</span>
                            <div><h2>Nhịp vận hành</h2><p>Theo dõi tài nguyên của tiến trình hiện tại theo thời gian thực.</p></div>
                        </div>""")
                        refresh_monitor_btn = gr.Button("Cập nhật số liệu", variant="primary")
                        with gr.Row(elem_classes="metrics-grid"):
                            with gr.Column(elem_classes="metric-card"):
                                gr.Markdown("Mức sử dụng CPU", elem_classes="metric-title")
                                cpu_value = gr.Markdown("Đang tải...", elem_classes="metric-value")
                                cpu_progress = gr.HTML('<div class="progress-container"><div class="progress-bar" style="transform:scaleX(0)"></div></div>')
                                cpu_info = gr.Markdown("Số lõi: đang tải...", elem_classes="metric-trend")
                            with gr.Column(elem_classes="metric-card"):
                                gr.Markdown("Mức sử dụng bộ nhớ", elem_classes="metric-title")
                                memory_value = gr.Markdown("Đang tải...", elem_classes="metric-value")
                                memory_progress = gr.HTML('<div class="progress-container"><div class="progress-bar" style="transform:scaleX(0)"></div></div>')
                                memory_info = gr.Markdown("Tổng bộ nhớ: đang tải...", elem_classes="metric-trend")
                            with gr.Column(elem_classes="metric-card"):
                                gr.Markdown("Dung lượng ổ đĩa", elem_classes="metric-title")
                                disk_value = gr.Markdown("Đang tải...", elem_classes="metric-value")
                                disk_progress = gr.HTML('<div class="progress-container"><div class="progress-bar" style="transform:scaleX(0)"></div></div>')
                                disk_info = gr.Markdown("Tổng dung lượng: đang tải...", elem_classes="metric-trend")
                            with gr.Column(elem_classes="metric-card"):
                                gr.Markdown("Kho vector", elem_classes="metric-title")
                                vector_db_value = gr.Markdown("Phân đoạn: 0", elem_classes="metric-value")
                                vector_db_info = gr.Markdown("Vector: 0", elem_classes="metric-trend")

                with gr.Group(elem_classes=["bezel-shell", "reveal"]):
                    with gr.Group(elem_classes="bezel-core"):
                        gr.HTML("""<div class="section-heading">
                            <span class="section-number">02</span>
                            <div><h2>Nhật ký hệ thống</h2><p>Thông báo vận hành gần nhất của ứng dụng.</p></div>
                        </div>""")
                        with gr.Row():
                            log_level = gr.Dropdown(
                                choices=["Tất cả", "Thông tin", "Cảnh báo", "Lỗi"],
                                value="Tất cả",
                                label="Mức nhật ký",
                            )
                            clear_logs_btn = gr.Button("Xóa nhật ký", variant="secondary")
                        log_display = gr.HTML("", elem_classes="log-container")

    # Hàm xử lý sự kiện
    def clear_chat_history():
        return [], "Đã xóa cuộc trò chuyện."

    def process_chat(question, history, enable_web_search, model_choice_val):
        if history is None or not isinstance(history, list):
            history = []

        api_text = build_api_info_html(
            enable_web_search,
            model_choice_val,
        )

        if not question or question.strip() == "":
            history.append(
                {
                    "role": "assistant",
                    "content": "Vui lòng nhập câu hỏi trước khi gửi.",
                }
            )
            return history, "", api_text

        try:
            answer = query_answer(question, enable_web_search, model_choice_val)
        except Exception as e:
            answer = f"Không thể tạo câu trả lời: {e}"
            logging.error("Quá trình hỏi đáp gặp lỗi: %s", e)

        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": answer})
        return history, "", api_text

    def update_api_info(enable_web_search, model_choice_val):
        return build_api_info_html(
            enable_web_search,
            model_choice_val,
        )

    def get_system_metrics():
        """Lấy số liệu tài nguyên hệ thống."""
        try:
            import psutil
            cpu_pct = psutil.cpu_percent(interval=1)
            cpu_cnt = psutil.cpu_count(logical=False)
            mem = psutil.virtual_memory()
            mem_total = round(mem.total / (1024 ** 3), 1)
            mem_used = round(mem.used / (1024 ** 3), 1)
            disk = psutil.disk_usage('/')
            disk_total = round(disk.total / (1024 ** 3), 1)
            disk_used = round(disk.used / (1024 ** 3), 1)

            doc_count = len(vector_store.contents_map)
            vec_count = vector_store.total_chunks

            def bar(pct, color="var(--app-accent)"):
                scale = max(0.0, min(100.0, float(pct))) / 100
                return (
                    '<div class="progress-container">'
                    f'<div class="progress-bar" style="transform:scaleX({scale});background:{color}"></div>'
                    '</div>'
                )

            c_color = "#4CAF50" if cpu_pct < 50 else "#FFC107" if cpu_pct < 80 else "#f44336"
            m_color = "#4CAF50" if mem.percent < 50 else "#FFC107" if mem.percent < 80 else "#f44336"
            d_color = "#4CAF50" if disk.percent < 50 else "#FFC107" if disk.percent < 80 else "#f44336"

            now = datetime.now().strftime("%H:%M:%S")
            log = f'<div class="log-entry"><span style="color:var(--app-accent)">[{now}]</span> <span style="color:var(--app-success)">[THÔNG TIN]</span> Đã cập nhật số liệu hệ thống</div>'

            return (
                f"{cpu_pct}%", bar(cpu_pct, c_color), f"Lõi vật lý: {cpu_cnt}",
                f"{mem_used} GB / {mem_total} GB", bar(mem.percent, m_color), f"Đã dùng: {mem.percent}%",
                f"{disk_used} GB / {disk_total} GB", bar(disk.percent, d_color), f"Đã dùng: {disk.percent}%",
                f"Phân đoạn: {doc_count}", f"Vector: {vec_count}", log
            )
        except Exception as e:
            err = f"Không thể đọc số liệu hệ thống: {e}"
            return ("Lỗi", "", err, "Lỗi", "", err, "Lỗi", "", err, "Lỗi", err,
                    f"<div style='color:#f44336'>[LỖI] {err}</div>")

    # Liên kết sự kiện
    upload_btn.click(process_multiple_files, inputs=[file_input], outputs=[upload_status, file_list], show_progress=True)
    ask_btn.click(process_chat, inputs=[question_input, chatbot, web_search_checkbox, model_choice],
                  outputs=[chatbot, question_input, api_info])
    question_input.submit(
        process_chat,
        inputs=[question_input, chatbot, web_search_checkbox, model_choice],
        outputs=[chatbot, question_input, api_info],
    )
    clear_btn.click(clear_chat_history, inputs=[], outputs=[chatbot, status_display])
    web_search_checkbox.change(update_api_info, inputs=[web_search_checkbox, model_choice], outputs=[api_info])
    model_choice.change(update_api_info, inputs=[web_search_checkbox, model_choice], outputs=[api_info])
    refresh_chunks_btn.click(fn=get_document_chunks, outputs=[chunks_data, chunks_status])
    chunks_data.select(fn=show_chunk_details, outputs=chunk_detail_text)
    refresh_monitor_btn.click(fn=get_system_metrics, outputs=[
        cpu_value, cpu_progress, cpu_info,
        memory_value, memory_progress, memory_info,
        disk_value, disk_progress, disk_info,
        vector_db_value, vector_db_info, log_display
    ])
    clear_logs_btn.click(
        fn=lambda: "<div style='color:#4CAF50'>Đã xóa nhật ký.</div>",
        outputs=[log_display],
    )
    theme_btn.click(fn=toggle_theme, inputs=[], outputs=[], js="""
        () => {
            document.querySelector('body').classList.toggle('dark');
            const isDark = document.querySelector('body').classList.contains('dark');
            localStorage.setItem('rag-theme', isDark ? 'dark' : 'light');
        }
    """)


def check_environment():
    """Kiểm tra provider LLM API đã chọn và API key tương ứng."""
    provider = get_provider_name()
    provider_config = get_provider_config(provider)
    if not is_configured_api_key(provider_config.api_key):
        print(f"Chưa cấu hình API key cho provider {provider}.")
        print("   Hãy cập nhật file .env rồi khởi động lại ứng dụng.")
        return False

    print(f"Đã cấu hình API key cho provider {provider}.")
    result = call_llm(
        "Chỉ trả lời đúng hai từ: kết nối thành công",
        provider=provider,
        temperature=0.1,
        max_tokens=256,
    )
    if result.startswith(("Lỗi", "Không thể", "Phản hồi")):
        print(f"Kiểm tra kết nối {provider} thất bại: {result}")
        return False
    print(f"Kết nối {provider} thành công.")
    return True


if __name__ == "__main__":
    if not check_environment():
        exit(1)

    ports = [17995, 17996, 17997, 17998, 17999]
    selected_port = next((p for p in ports if is_port_available(p)), None)

    if not selected_port:
        print("Không còn cổng khả dụng trong dải 17995–17999.")
        exit(1)

    try:
        webbrowser.open(f"http://127.0.0.1:{selected_port}")
        demo.launch(
            server_port=selected_port, server_name="0.0.0.0",
            show_error=True, ssl_verify=False, height=900,
            theme=APP_THEME, css=CSS, js=THEME_JS,
        )
    except Exception as e:
        print(f"Không thể khởi động ứng dụng: {e}")
