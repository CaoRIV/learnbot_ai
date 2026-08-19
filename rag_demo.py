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
        "Mô hình embedding": "all-MiniLM-L6-v2",
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
        gr.themes.GoogleFont("Be Vietnam Pro", weights=(400, 500, 600, 700)),
        "Segoe UI",
        "sans-serif",
    ],
    font_mono=["Cascadia Code", "Consolas", "monospace"],
    radius_size=gr.themes.sizes.radius_lg,
    spacing_size=gr.themes.sizes.spacing_md,
).set(
    body_background_fill="#f5f5f1",
    body_background_fill_dark="#111310",
    body_text_color="#171717",
    body_text_color_dark="#f5f5f1",
    body_text_color_subdued="#57534e",
    body_text_color_subdued_dark="#b7b3aa",
    background_fill_primary="#ffffff",
    background_fill_primary_dark="#191b18",
    background_fill_secondary="#f8f8f5",
    background_fill_secondary_dark="#20231f",
    block_background_fill="#ffffff",
    block_background_fill_dark="#191b18",
    block_border_color="#e7e5df",
    block_border_color_dark="#343731",
    block_radius="18px",
    block_shadow="none",
    input_background_fill="#ffffff",
    input_background_fill_dark="#151714",
    input_border_color="#d8d5cd",
    input_border_color_dark="#44483f",
    input_border_color_focus="#b88920",
    input_border_color_focus_dark="#e2bd61",
    input_radius="14px",
    button_primary_background_fill="#c99b2e",
    button_primary_background_fill_dark="#e0b955",
    button_primary_background_fill_hover="#b48724",
    button_primary_background_fill_hover_dark="#edc96d",
    button_primary_text_color="#171717",
    button_primary_text_color_dark="#171717",
    button_primary_border_color="#c99b2e",
    button_primary_border_color_dark="#e0b955",
    button_primary_shadow="none",
    button_primary_shadow_hover="0 8px 20px rgba(159, 113, 12, 0.20)",
    button_secondary_background_fill="#ffffff",
    button_secondary_background_fill_dark="#20231f",
    button_secondary_border_color="#d8d5cd",
    button_secondary_border_color_dark="#44483f",
    button_secondary_text_color="#292524",
    button_secondary_text_color_dark="#f5f5f1",
    button_transform_hover="none",
    button_transition="background-color 160ms ease, border-color 160ms ease, box-shadow 160ms ease",
)

CSS = """
:root {
    --app-bg:#f5f5f1;
    --app-surface:#ffffff;
    --app-surface-muted:#f8f8f5;
    --app-border:#e4e1d9;
    --app-text:#171717;
    --app-muted:#625f58;
    --app-accent:#c99b2e;
    --app-accent-soft:#faf2dd;
    --app-success:#217a52;
    --app-shadow:0 18px 48px rgba(36,31,20,.07);
}
body.dark {
    --app-bg:#111310;
    --app-surface:#191b18;
    --app-surface-muted:#20231f;
    --app-border:#343731;
    --app-text:#f5f5f1;
    --app-muted:#b7b3aa;
    --app-accent:#e0b955;
    --app-accent-soft:#322a18;
    --app-success:#69c99b;
    --app-shadow:0 20px 50px rgba(0,0,0,.20);
}
html { scroll-behavior:smooth; }
body { background:var(--app-bg)!important; }
.gradio-container {
    max-width:1440px!important;
    width:100%!important;
    margin:0 auto!important;
    padding:24px clamp(16px,3vw,44px) 40px!important;
    overflow-x:hidden;
}
footer { display:none!important; }
button, [role="button"], [role="tab"] { cursor:pointer!important; }
button:focus-visible, [role="button"]:focus-visible, [role="tab"]:focus-visible,
textarea:focus-visible, input:focus-visible {
    outline:3px solid color-mix(in srgb, var(--app-accent) 55%, transparent)!important;
    outline-offset:2px!important;
}
.topbar { align-items:center!important; margin-bottom:18px!important; }
.brand-shell { display:flex; align-items:flex-start; gap:16px; padding:4px 0; }
.brand-mark {
    display:grid; place-items:center; flex:0 0 48px; width:48px; height:48px;
    border-radius:15px; color:#171717; background:var(--app-accent);
    box-shadow:0 10px 24px rgba(159,113,12,.18);
}
.brand-mark svg { width:25px; height:25px; }
.brand-eyebrow {
    margin:0 0 4px; color:var(--app-muted); font-size:12px; font-weight:700;
    letter-spacing:.12em; text-transform:uppercase;
}
.brand-title {
    margin:0; color:var(--app-text); font-size:clamp(24px,3vw,36px);
    line-height:1.12; letter-spacing:-.035em; font-weight:700;
}
.brand-copy { margin:8px 0 0; max-width:720px; color:var(--app-muted); font-size:14px; line-height:1.6; }
.trust-row { display:flex; flex-wrap:wrap; gap:8px; margin-top:12px; }
.trust-chip {
    display:inline-flex; align-items:center; min-height:28px; padding:4px 10px;
    border:1px solid var(--app-border); border-radius:999px; color:var(--app-muted);
    background:var(--app-surface); font-size:11px; font-weight:600;
}
.theme-toggle-btn {
    min-width:112px!important; min-height:40px!important; padding:8px 14px!important;
    border-radius:999px!important; font-size:12px!important; font-weight:600!important;
}
.main-tabs > .tab-nav, .main-tabs [role="tablist"] {
    gap:6px!important; padding:5px!important; border:1px solid var(--app-border)!important;
    border-radius:14px!important; background:var(--app-surface)!important;
}
.main-tabs [role="tab"] {
    min-height:38px!important; padding:8px 14px!important; border-radius:10px!important;
    color:var(--app-muted)!important; font-size:13px!important; font-weight:600!important;
    transition:background-color 160ms ease,color 160ms ease!important;
}
.main-tabs [role="tab"][aria-selected="true"] {
    color:var(--app-text)!important; background:var(--app-accent-soft)!important;
}
.workspace-grid { align-items:flex-start!important; gap:18px!important; padding-top:18px!important; }
.surface-card {
    padding:20px!important; border:1px solid var(--app-border)!important;
    border-radius:22px!important; background:var(--app-surface)!important;
    box-shadow:var(--app-shadow)!important;
}
.document-panel { position:sticky; top:18px; }
.section-heading { display:flex; align-items:flex-start; gap:12px; margin:0 0 16px; }
.section-number {
    display:grid; place-items:center; flex:0 0 30px; width:30px; height:30px;
    border-radius:9px; background:var(--app-accent-soft); color:var(--app-text);
    font-size:11px; font-weight:700;
}
.section-heading h2 { margin:0; color:var(--app-text); font-size:17px; line-height:1.3; font-weight:700; }
.section-heading p { margin:4px 0 0; color:var(--app-muted); font-size:12px; line-height:1.5; }
.format-note {
    margin:0 0 10px!important; padding:0!important; overflow:visible!important;
    color:var(--app-muted)!important; font-size:11px!important; line-height:1.5!important;
}
#upload-zone {
    min-height:168px!important; border:1px dashed #aaa397!important;
    border-radius:16px!important; background:var(--app-surface-muted)!important;
    transition:border-color 160ms ease,background-color 160ms ease!important;
}
#upload-zone:hover { border-color:var(--app-accent)!important; background:var(--app-accent-soft)!important; }
.upload-action { min-height:44px!important; margin-top:10px!important; }
.process-details { margin-top:12px!important; border-color:var(--app-border)!important; border-radius:14px!important; }
.file-list { margin-top:8px!important; }
.chat-container {
    min-height:500px!important; border:1px solid var(--app-border)!important;
    border-radius:18px!important; background:var(--app-surface-muted)!important;
}
.composer-card {
    margin-top:12px!important; padding:12px!important; border:1px solid var(--app-border)!important;
    border-radius:18px!important; background:var(--app-surface)!important;
    box-shadow:0 10px 28px rgba(36,31,20,.06)!important;
}
.composer-card > .composer-card,
.model-card > .model-card,
.monitor-panel > .monitor-panel {
    padding:0!important; border:0!important; border-radius:0!important;
    background:transparent!important; box-shadow:none!important;
}
.model-card .styler, .monitor-panel .styler { background:transparent!important; }
.composer-card textarea { line-height:1.55!important; }
.composer-options { align-items:end!important; gap:10px!important; }
.send-button, .clear-button { min-height:42px!important; }
.api-info {
    display:flex; flex-wrap:wrap; gap:8px; margin-top:10px; color:var(--app-muted);
    font-size:11px;
}
.api-badge {
    display:inline-flex; align-items:center; min-height:26px; padding:4px 9px;
    border:1px solid var(--app-border); border-radius:999px; background:var(--app-surface-muted);
}
.api-badge strong { margin-left:4px; color:var(--app-text); }
.footer-note { margin:10px 2px 0; color:var(--app-muted); font-size:11px; line-height:1.55; }
.chunk-layout { padding-top:18px!important; gap:18px!important; }
.model-card, .monitor-panel {
    padding:18px!important; border:1px solid var(--app-border)!important;
    border-radius:18px!important; background:var(--app-surface)!important;
}
.chunk-table { border-radius:16px!important; overflow:hidden!important; }
.chunk-detail-box { min-height:200px; font-family:"Cascadia Code",Consolas,monospace; white-space:pre-wrap; }
.metrics-grid { gap:12px!important; }
.metric-card {
    min-width:180px!important; padding:16px!important; border:1px solid var(--app-border)!important;
    border-radius:16px!important; background:var(--app-surface-muted)!important;
}
.metric-title { margin-bottom:8px!important; color:var(--app-muted)!important; font-size:12px!important; }
.metric-value { margin-bottom:3px!important; color:var(--app-text)!important; font-size:22px!important; font-weight:700!important; }
.metric-trend { color:var(--app-success)!important; font-size:11px!important; }
.progress-container { width:100%; margin:10px 0; overflow:hidden; border-radius:999px; background:var(--app-border); }
.progress-bar { height:7px; border-radius:999px; background:var(--app-accent); transition:width 240ms ease; }
.log-container {
    max-height:300px; overflow-y:auto; padding:14px; border:1px solid var(--app-border);
    border-radius:14px; background:var(--app-surface-muted); font-family:"Cascadia Code",Consolas,monospace; font-size:12px;
}
@media (max-width: 900px) {
    .gradio-container { padding:14px 12px 28px!important; }
    .topbar { gap:10px!important; }
    .document-panel { position:static; }
    .surface-card { padding:16px!important; border-radius:18px!important; }
    .chat-container { min-height:420px!important; height:420px!important; }
    .metrics-grid { flex-wrap:wrap!important; }
    .metric-card { flex:1 1 44%!important; }
}
@media (max-width: 560px) {
    .brand-shell { gap:11px; }
    .brand-mark { flex-basis:42px; width:42px; height:42px; border-radius:13px; }
    .brand-title { font-size:24px; }
    .brand-copy { font-size:12px; }
    .trust-row { display:none; }
    .theme-toggle-btn { width:100%!important; }
    .main-tabs [role="tab"] { padding:8px 9px!important; font-size:12px!important; }
    .composer-options, .composer-actions { flex-direction:column!important; align-items:stretch!important; }
    .composer-options > *, .composer-actions > * {
        width:100%!important; min-width:0!important; flex:1 1 auto!important;
    }
    .metric-card { flex-basis:100%!important; }
    .chat-container { min-height:360px!important; height:360px!important; }
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
    with gr.Row(elem_classes="topbar"):
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
                        <p class="brand-eyebrow">learnbot_ai · RAG tiếng Việt</p>
                        <h1 class="brand-title">Hỏi tài liệu, nhận câu trả lời có căn cứ</h1>
                        <p class="brand-copy">Tạo kho tri thức từ tài liệu của bạn và truy xuất câu trả lời kèm nguồn, số trang rõ ràng.</p>
                        <div class="trust-row" aria-label="Năng lực chính">
                            <span class="trust-chip">Trích dẫn theo trang</span>
                            <span class="trust-chip">FAISS + BM25</span>
                            <span class="trust-chip">LLM qua API</span>
                        </div>
                    </div>
                </div>
            """)
        with gr.Column(scale=2, min_width=130):
            theme_btn = gr.Button(
                "Sáng / tối",
                min_width=112,
                elem_classes="theme-toggle-btn",
            )

    with gr.Tabs(elem_classes="main-tabs") as tabs:
        # Thẻ hỏi đáp
        with gr.TabItem("Trò chuyện"):
            with gr.Row(equal_height=False, elem_classes="workspace-grid"):
                with gr.Column(
                    scale=4,
                    min_width=300,
                    elem_classes=["surface-card", "document-panel"],
                ):
                    gr.HTML("""<div class="section-heading">
                        <span class="section-number">01</span>
                        <div><h2>Chuẩn bị kho tài liệu</h2><p>Tải tệp lên và lập chỉ mục trước khi đặt câu hỏi.</p></div>
                    </div>""")
                    gr.HTML(
                        "<p class=\"format-note\">PDF · Word · Excel · PowerPoint · TXT · Markdown</p>"
                    )
                    file_input = gr.File(
                        label="Tài liệu nguồn",
                        file_types=[".pdf", ".txt", ".docx", ".xlsx", ".xls", ".pptx", ".md"],
                        file_count="multiple",
                        height=168,
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
                    elem_classes=["surface-card", "conversation-panel"],
                ):
                    gr.HTML("""<div class="section-heading">
                        <span class="section-number">02</span>
                        <div><h2>Tra cứu nội dung</h2><p>Câu trả lời chỉ dựa trên nguồn đã truy xuất.</p></div>
                    </div>""")
                    chatbot = gr.Chatbot(
                        label="Lịch sử trò chuyện",
                        height=500,
                        elem_classes="chat-container",
                        show_label=False,
                        layout="bubble",
                        buttons=["copy"],
                        placeholder="Tải và xử lý tài liệu, sau đó đặt câu hỏi đầu tiên của bạn.",
                    )
                    api_info = gr.HTML(
                        build_api_info_html(False, DEFAULT_MODEL_CHOICE)
                    )
                    with gr.Group(elem_classes="composer-card"):
                        question_input = gr.Textbox(
                            label="Câu hỏi về tài liệu",
                            lines=2,
                            max_lines=6,
                            placeholder="Ví dụ: Tóm tắt nội dung chính và dẫn nguồn theo từng trang...",
                            autofocus=True,
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
                        Nhấn Enter để gửi, Shift + Enter để xuống dòng. Hãy đối chiếu trích dẫn khi dùng thông tin quan trọng.
                    </div>""")

        # Thẻ xem phân đoạn
        with gr.TabItem("Phân đoạn"):
            with gr.Row(elem_classes="chunk-layout"):
                with gr.Column(
                    scale=1,
                    min_width=280,
                    elem_classes="surface-card",
                ):
                    gr.HTML("""<div class="section-heading">
                        <span class="section-number">A</span>
                        <div><h2>Cấu hình truy xuất</h2><p>Mô hình và kỹ thuật đang được sử dụng.</p></div>
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
                    elem_classes="surface-card",
                ):
                    gr.HTML("""<div class="section-heading">
                        <span class="section-number">B</span>
                        <div><h2>Dữ liệu đã lập chỉ mục</h2><p>Kiểm tra nội dung trước khi hệ thống truy xuất.</p></div>
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
            with gr.Column():
                with gr.Group(elem_classes="monitor-panel"):
                    gr.HTML("""<div class="section-heading">
                        <span class="section-number">01</span>
                        <div><h2>Tài nguyên hệ thống</h2><p>Theo dõi mức sử dụng của tiến trình hiện tại.</p></div>
                    </div>""")
                    refresh_monitor_btn = gr.Button("Cập nhật số liệu", variant="primary")
                    with gr.Row(elem_classes="metrics-grid"):
                        with gr.Column(elem_classes="metric-card"):
                            gr.Markdown("Mức sử dụng CPU", elem_classes="metric-title")
                            cpu_value = gr.Markdown("Đang tải...", elem_classes="metric-value")
                            cpu_progress = gr.HTML('<div class="progress-container"><div class="progress-bar" style="width:0%"></div></div>')
                            cpu_info = gr.Markdown("Số lõi: đang tải...", elem_classes="metric-trend")
                        with gr.Column(elem_classes="metric-card"):
                            gr.Markdown("Mức sử dụng bộ nhớ", elem_classes="metric-title")
                            memory_value = gr.Markdown("Đang tải...", elem_classes="metric-value")
                            memory_progress = gr.HTML('<div class="progress-container"><div class="progress-bar" style="width:0%"></div></div>')
                            memory_info = gr.Markdown("Tổng bộ nhớ: đang tải...", elem_classes="metric-trend")
                        with gr.Column(elem_classes="metric-card"):
                            gr.Markdown("Dung lượng ổ đĩa", elem_classes="metric-title")
                            disk_value = gr.Markdown("Đang tải...", elem_classes="metric-value")
                            disk_progress = gr.HTML('<div class="progress-container"><div class="progress-bar" style="width:0%"></div></div>')
                            disk_info = gr.Markdown("Tổng dung lượng: đang tải...", elem_classes="metric-trend")
                        with gr.Column(elem_classes="metric-card"):
                            gr.Markdown("Kho vector", elem_classes="metric-title")
                            vector_db_value = gr.Markdown("Phân đoạn: 0", elem_classes="metric-value")
                            vector_db_info = gr.Markdown("Vector: 0", elem_classes="metric-trend")

                with gr.Group(elem_classes="monitor-panel"):
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

            def bar(pct, color="var(--tech-cyan)"):
                return f'<div class="progress-container"><div class="progress-bar" style="width:{pct}%;background:{color}"></div></div>'

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
