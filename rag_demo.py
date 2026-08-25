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
from core.index_snapshot import restore_indexes
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
    body_background_fill="#f5f4f0",
    body_background_fill_dark="#171716",
    body_text_color="#141413",
    body_text_color_dark="#faf9f5",
    body_text_color_subdued="#68665f",
    body_text_color_subdued_dark="#b0aea5",
    background_fill_primary="#fbfaf7",
    background_fill_primary_dark="#1d1d1b",
    background_fill_secondary="#f0efea",
    background_fill_secondary_dark="#242421",
    block_background_fill="#fbfaf7",
    block_background_fill_dark="#1d1d1b",
    block_border_color="rgba(20,20,19,0.08)",
    block_border_color_dark="rgba(250,249,245,0.10)",
    block_radius="14px",
    block_shadow="none",
    input_background_fill="#fbfaf7",
    input_background_fill_dark="#181816",
    input_border_color="rgba(20,20,19,0.12)",
    input_border_color_dark="rgba(250,249,245,0.14)",
    input_border_color_focus="#d97757",
    input_border_color_focus_dark="#d97757",
    input_radius="12px",
    button_primary_background_fill="#c96d4f",
    button_primary_background_fill_dark="#d77b5d",
    button_primary_background_fill_hover="#b96044",
    button_primary_background_fill_hover_dark="#e08769",
    button_primary_text_color="#141413",
    button_primary_text_color_dark="#141413",
    button_primary_border_color="#c96d4f",
    button_primary_border_color_dark="#d77b5d",
    button_primary_shadow="none",
    button_primary_shadow_hover="none",
    button_secondary_background_fill="#efeee9",
    button_secondary_background_fill_dark="#242421",
    button_secondary_border_color="rgba(20,20,19,0.08)",
    button_secondary_border_color_dark="rgba(250,249,245,0.10)",
    button_secondary_text_color="#141413",
    button_secondary_text_color_dark="#faf9f5",
    button_transform_hover="none",
    button_transition="transform 180ms ease, background-color 180ms ease, border-color 180ms ease",
)

CSS = """
:root {
    --app-bg:#f5f4f0;
    --app-surface:#fbfaf7;
    --app-surface-muted:#f0efea;
    --app-shell:#e7e5df;
    --app-text:#141413;
    --app-muted:#68665f;
    --app-mid:#b0aea5;
    --app-accent:#c96d4f;
    --app-accent-strong:#a95137;
    --app-accent-soft:#f1dfd7;
    --app-success:#788c5d;
    --app-danger:#b95446;
    --hairline:rgba(20,20,19,.09);
    --app-shadow:none;
    --app-shadow-raised:none;
    --motion:cubic-bezier(.16,1,.3,1);
    --font-display:"Segoe UI Variable Display","Segoe UI",sans-serif;
    --font-body:"Segoe UI Variable Text","Segoe UI",sans-serif;
    --font-mono:"Cascadia Code",Consolas,monospace;
}
body.dark {
    --app-bg:#171716;
    --app-surface:#1d1d1b;
    --app-surface-muted:#22221f;
    --app-shell:#292824;
    --app-text:#faf9f5;
    --app-muted:#b0aea5;
    --app-mid:#79776f;
    --app-accent:#d77b5d;
    --app-accent-strong:#e18a6e;
    --app-accent-soft:#3a241d;
    --app-success:#91a876;
    --app-danger:#df806f;
    --hairline:rgba(250,249,245,.11);
    --app-shadow:none;
    --app-shadow-raised:none;
}
html { scroll-behavior:smooth; }
body {
    background:var(--app-bg)!important;
    color:var(--app-text)!important;
    font-family:var(--font-body)!important;
}
.gradio-container {
    max-width:1440px!important;
    width:100%!important;
    margin:0 auto!important;
    padding:16px clamp(16px,3vw,40px) 24px!important;
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
    cursor:pointer!important; transition:transform 180ms ease,background-color 180ms ease,color 180ms ease,border-color 180ms ease!important;
}
button:hover, [role="button"]:hover { transform:none; }
button:active, [role="button"]:active { transform:scale(.98); }
button:focus-visible, [role="button"]:focus-visible, [role="tab"]:focus-visible,
textarea:focus-visible, input:focus-visible {
    outline:3px solid color-mix(in srgb,var(--app-accent) 52%,transparent)!important;
    outline-offset:3px!important;
}
.topbar {
    width:100%!important; min-height:60px!important; align-items:center!important;
    margin:0!important; padding:0 0 12px!important; border:0!important; border-bottom:1px solid var(--hairline)!important;
    border-radius:0!important; background:transparent!important; box-shadow:none!important;
}
.brand-shell { display:flex; align-items:center; gap:12px; padding:0; }
.brand-mark {
    display:grid; place-items:center; flex:0 0 38px; width:38px; height:38px;
    border-radius:10px; color:#141413; background:var(--app-accent); box-shadow:none;
}
.brand-mark svg { width:20px; height:20px; }
.brand-title {
    margin:0; color:var(--app-text); font-size:18px; line-height:1.2; letter-spacing:-.025em; font-weight:650;
}
.brand-copy { margin:3px 0 0; color:var(--app-muted); font-size:12px; line-height:1.35; }
.theme-toggle-btn {
    min-width:104px!important; min-height:38px!important; padding:7px 12px!important;
    border:1px solid var(--hairline)!important; border-radius:10px!important; font-size:12px!important; font-weight:600!important;
    letter-spacing:0!important; background:transparent!important; box-shadow:none!important;
}
.main-tabs > .tab-nav, .main-tabs [role="tablist"] {
    width:100%!important; max-width:100%!important; gap:24px!important; margin:0!important;
    padding:0!important; border:0!important; border-bottom:1px solid var(--hairline)!important;
    border-radius:0!important; background:transparent!important; box-shadow:none!important;
}
.main-tabs { width:100%!important; }
.main-tabs [role="tab"] {
    min-height:42px!important; padding:10px 2px!important; border:0!important; border-bottom:2px solid transparent!important;
    border-radius:0!important; color:var(--app-muted)!important; font-size:12px!important; font-weight:600!important; letter-spacing:0!important;
}
.main-tabs [role="tab"][aria-selected="true"] {
    color:var(--app-text)!important; background:transparent!important; border-bottom-color:var(--app-accent)!important; box-shadow:none!important;
}
.workspace-grid {
    align-items:stretch!important; gap:0!important; padding-top:16px!important;
    height:calc(100dvh - 146px); min-height:610px;
}
.bezel-shell {
    padding:0!important; border:0!important; border-radius:0!important; background:transparent!important; box-shadow:none!important;
}
.bezel-core {
    padding:20px!important; border:0!important; border-radius:0!important; background:transparent!important; box-shadow:none!important;
}
.bezel-core > .bezel-core { padding:0!important; background:transparent!important; box-shadow:none!important; }
.document-panel { position:static; border-right:1px solid var(--hairline)!important; }
.document-panel > .bezel-core { height:100%!important; overflow-y:auto; padding-left:0!important; padding-right:20px!important; }
.conversation-panel { margin-top:0!important; min-width:0!important; }
.conversation-panel > .bezel-core { height:100%!important; min-height:0!important; padding:0 0 0 24px!important; display:flex!important; flex-direction:column!important; }
.section-heading { display:block; margin:0 0 16px; }
.section-heading h2 { margin:0; color:var(--app-text); font-size:15px; line-height:1.3; letter-spacing:-.01em; font-weight:650; }
.section-heading p { margin:4px 0 0; max-width:52ch; color:var(--app-muted); font-size:12px; line-height:1.55; }
.format-note {
    margin:0 0 10px!important; padding:0!important; overflow:visible!important;
    color:var(--app-muted)!important; font-family:var(--font-mono)!important; font-size:9px!important;
    line-height:1.5!important; letter-spacing:.04em!important; text-transform:none;
}
#upload-zone {
    min-height:154px!important; border:1px dashed color-mix(in srgb,var(--app-muted) 45%,transparent)!important;
    border-radius:12px!important; background:transparent!important; box-shadow:none!important;
    transition:border-color 180ms ease,background-color 180ms ease!important;
}
#upload-zone:hover { transform:none; border-color:var(--app-accent)!important; background:color-mix(in srgb,var(--app-accent-soft) 35%,transparent)!important; box-shadow:none!important; }
.upload-action,.send-button {
    min-height:42px!important; margin-top:10px!important; border:1px solid transparent!important; border-radius:10px!important;
    color:#141413!important; box-shadow:none!important; font-size:12px!important; letter-spacing:0!important;
}
.process-details,.composer-settings {
    margin-top:10px!important; border:1px solid var(--hairline)!important; border-radius:10px!important;
    background:transparent!important; box-shadow:none!important;
}
.file-list { margin-top:8px!important; }
.chat-container {
    flex:1 1 0!important; min-height:300px!important; height:auto!important;
    border:1px solid var(--hairline)!important; border-radius:14px!important; background:var(--app-surface)!important; box-shadow:none!important;
}
.chat-container .message {
    border:0!important; border-radius:12px!important; box-shadow:none!important;
}
.chat-container .message.user { background:var(--app-accent-soft)!important; }
.chat-container .message.bot { background:var(--app-surface-muted)!important; }
.chat-container .message p { line-height:1.65!important; }
.composer-card {
    margin-top:10px!important; padding:0!important; border:1px solid var(--hairline)!important; border-radius:14px!important;
    background:var(--app-surface)!important; box-shadow:none!important;
}
.composer-card > .composer-card {
    padding:12px!important; border:0!important; border-radius:13px!important; background:transparent!important; box-shadow:none!important;
}
.composer-card textarea { line-height:1.55!important; }
.composer-options { align-items:end!important; gap:12px!important; }
.composer-actions { align-items:center!important; gap:10px!important; }
.clear-button { min-height:42px!important; border:1px solid var(--hairline)!important; border-radius:10px!important; background:transparent!important; }
.api-info {
    display:flex; flex-wrap:wrap; gap:0; margin:7px 2px 0; color:var(--app-muted); font-size:10px;
}
.api-badge {
    display:inline-flex; align-items:center; min-height:24px; padding:2px 0; border-radius:0; background:transparent; box-shadow:none;
}
.api-badge + .api-badge::before { content:""; width:1px; height:12px; margin:0 10px; border-radius:0; background:var(--hairline); }
.api-badge strong { margin-left:4px; color:var(--app-text); }
.footer-note {
    display:block; margin:7px 0 0; padding:0 2px; overflow:visible;
    color:var(--app-muted); font-size:10px; line-height:1.65;
}
.chunk-layout { align-items:flex-start!important; padding-top:18px!important; gap:0!important; }
.model-card,.monitor-core {
    padding:0!important; border:0!important; border-radius:0!important; background:transparent!important; box-shadow:none!important;
}
.model-card > .model-card,.monitor-core > .monitor-core { padding:0!important; background:transparent!important; box-shadow:none!important; }
.chunk-table { border-radius:16px!important; overflow:hidden!important; }
.chunk-detail-box { min-height:200px; font-family:"Cascadia Code",Consolas,monospace; white-space:pre-wrap; }
.system-stack { gap:0!important; padding-top:18px!important; }
.system-stack > * + * { border-top:1px solid var(--hairline)!important; margin-top:24px!important; padding-top:24px!important; }
.metrics-grid { gap:0!important; border-block:1px solid var(--hairline)!important; }
.metric-card {
    min-width:180px!important; padding:18px!important; border:0!important; border-radius:0!important;
    background:transparent!important; box-shadow:none!important; transition:background-color 180ms ease!important;
}
.metric-card + .metric-card { border-left:1px solid var(--hairline)!important; }
.metric-card:hover { transform:none; background:var(--app-surface-muted)!important; }
.metric-title { margin-bottom:12px!important; color:var(--app-muted)!important; font-size:10px!important; letter-spacing:.06em!important; }
.metric-value { margin-bottom:3px!important; color:var(--app-text)!important; font-size:clamp(23px,2.4vw,34px)!important; font-weight:600!important; font-variant-numeric:tabular-nums; }
.metric-trend { color:var(--app-success)!important; font-size:11px!important; }
.progress-container { width:100%; height:6px; margin:13px 0; overflow:hidden; border-radius:999px; background:var(--app-shell); }
.progress-bar { width:100%; height:6px; transform:scaleX(0); transform-origin:left center; border-radius:999px; background:var(--app-accent); transition:transform 300ms var(--motion); }
.log-container {
    min-height:116px; max-height:300px; overflow-y:auto; padding:16px; border:1px solid var(--hairline); border-radius:10px;
    background:var(--app-surface); box-shadow:none;
    font-family:"Cascadia Code",Consolas,monospace; font-size:11px;
}
.reveal,body.ui-ready .reveal,body.ui-ready .reveal.is-visible { opacity:1; transform:none; filter:none; }
@media (max-width: 900px) {
    .gradio-container { padding:12px 14px 24px!important; }
    .topbar { gap:10px!important; }
    .workspace-grid,.chunk-layout { flex-direction:column!important; }
    .workspace-grid > *, .chunk-layout > * { width:100%!important; min-width:0!important; }
    .workspace-grid { height:auto; min-height:0; gap:24px!important; }
    .document-panel,.conversation-panel { position:static; margin-top:0!important; transform:none; border:0!important; }
    .conversation-panel { order:1; }
    .document-panel { order:2; border-top:1px solid var(--hairline)!important; padding-top:20px!important; }
    .document-panel > .bezel-core,.conversation-panel > .bezel-core { height:auto!important; overflow:visible; padding:0!important; }
    .bezel-shell,.bezel-core { border-radius:0!important; }
    .chat-container { min-height:380px!important; height:380px!important; }
    .metrics-grid { flex-wrap:wrap!important; }
    .metric-card { flex:1 1 44%!important; }
    .metric-card:nth-child(odd) { border-left:0!important; }
}
@media (max-width: 560px) {
    .gradio-container { padding:10px 12px 20px!important; }
    .topbar,.main-tabs { width:100%!important; min-width:0!important; }
    .topbar { min-height:54px!important; margin:0!important; padding:0 0 10px!important; border-radius:0!important; }
    .brand-shell { gap:9px; }
    .brand-mark { flex-basis:34px; width:34px; height:34px; border-radius:9px; }
    .brand-title { font-size:16px; line-height:1.15; }
    .brand-copy { display:none; }
    .theme-toggle-btn { width:auto!important; min-width:86px!important; }
    .main-tabs > .tab-nav,.main-tabs [role="tablist"] { width:100%!important; justify-content:flex-start!important; gap:18px!important; }
    .main-tabs [role="tab"] { padding:8px 1px!important; font-size:11px!important; }
    .composer-options, .composer-actions { flex-direction:column!important; align-items:stretch!important; }
    .composer-options > *, .composer-actions > * {
        width:100%!important; min-width:0!important; flex:1 1 auto!important;
    }
    .metric-card { flex-basis:100%!important; }
    .chat-container { min-height:340px!important; height:340px!important; }
}
@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after { scroll-behavior:auto!important; transition-duration:0.01ms!important; animation-duration:0.01ms!important; }
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
    applySavedTheme();
    document.body.classList.add('ui-ready');
    window.setTimeout(translateUpload, 120);
    const observer = new MutationObserver(translateUpload);
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
        with gr.Column(scale=10, min_width=180):
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
                        <h1 class="brand-title">LearnBot</h1>
                        <p class="brand-copy">Trợ lý hỏi đáp tài liệu tiếng Việt</p>
                    </div>
                </div>
            """)
        with gr.Column(scale=2, min_width=90):
            theme_btn = gr.Button(
                "Sáng / tối",
                min_width=112,
                elem_classes="theme-toggle-btn",
            )

    with gr.Tabs(elem_id="noi-dung-chinh", elem_classes="main-tabs") as tabs:
        # Thẻ hỏi đáp
        with gr.TabItem("Hỏi đáp"):
            with gr.Row(equal_height=False, elem_classes="workspace-grid"):
                with gr.Column(
                    scale=4,
                    min_width=300,
                    elem_classes=["bezel-shell", "document-panel", "reveal"],
                ):
                    with gr.Group(elem_classes="bezel-core"):
                        gr.HTML("""<div class="section-heading">
                            <div><h2>Tài liệu</h2><p>Tải tệp và lập chỉ mục để bắt đầu hỏi đáp.</p></div>
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
                            "Lập chỉ mục",
                            variant="primary",
                            elem_classes="upload-action",
                        )
                        with gr.Accordion(
                            "Kết quả xử lý",
                            open=False,
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
                            <div><h2>Hỏi đáp</h2><p>Câu trả lời sử dụng nội dung đã truy xuất từ tài liệu.</p></div>
                        </div>""")
                        chatbot = gr.Chatbot(
                            label="Lịch sử trò chuyện",
                            height=520,
                            elem_classes="chat-container",
                            show_label=False,
                            layout="bubble",
                            buttons=["copy"],
                            placeholder="Chưa có cuộc trò chuyện. Tải tài liệu rồi nhập câu hỏi để bắt đầu.",
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
                            with gr.Accordion(
                                "Tùy chọn trả lời",
                                open=False,
                                elem_classes="composer-settings",
                            ):
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
                                    "Gửi",
                                    variant="primary",
                                    scale=2,
                                    elem_classes="send-button",
                                )
                                clear_btn = gr.Button(
                                    "Xóa",
                                    variant="secondary",
                                    elem_classes="clear-button",
                                    scale=1,
                                )
                        status_display = gr.HTML("")
                        gr.Markdown("""<div class="footer-note">
                            Enter để gửi · Shift + Enter để xuống dòng · Luôn kiểm tra trích dẫn trước khi dùng thông tin quan trọng.
                        </div>""")

        # Thẻ xem phân đoạn
        with gr.TabItem("Phân đoạn"):
            with gr.Row(elem_classes="chunk-layout"):
                with gr.Column(
                    scale=1,
                    min_width=280,
                    elem_classes=["bezel-shell", "reveal"],
                ):
                    with gr.Group(elem_classes="bezel-core"):
                        gr.HTML("""<div class="section-heading">
                            <div><h2>Cấu hình truy xuất</h2><p>Mô hình và kỹ thuật dùng để tìm nội dung.</p></div>
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
                            <div><h2>Phân đoạn đã lập chỉ mục</h2><p>Kiểm tra nội dung được sử dụng để trả lời.</p></div>
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
        with gr.TabItem("Hệ thống"):
            with gr.Column(elem_classes="system-stack"):
                with gr.Group(elem_classes=["bezel-shell", "reveal"]):
                    with gr.Group(elem_classes="bezel-core"):
                        gr.HTML("""<div class="section-heading">
                            <div><h2>Tài nguyên</h2><p>Số liệu của tiến trình đang chạy.</p></div>
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
                            <div><h2>Nhật ký</h2><p>Thông báo gần nhất của ứng dụng.</p></div>
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
    restore_result = restore_indexes()
    print(restore_result.message)

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
