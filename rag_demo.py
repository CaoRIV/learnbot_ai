"""Điểm khởi chạy giao diện Gradio của learnbot_ai.

Module này định nghĩa bố cục, liên kết sự kiện, điều phối xử lý tài liệu và
hiển thị số liệu hệ thống. Logic RAG chính nằm trong ``core`` và ``features``.
"""

import os
import time
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
from core.document_loader import extract_text_by_page
from core.text_splitter import split_text
from core.embeddings import encode_texts
from core.vector_store import vector_store
from core.bm25_index import bm25_manager, tokenize_vietnamese
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
    """Xử lý tài liệu: trích xuất → phân đoạn → embedding → lập chỉ mục."""
    if not files:
        return "Vui lòng chọn ít nhất một tài liệu để xử lý.", []

    try:
        progress(0.1, desc="Đang xóa dữ liệu cũ...")
        vector_store.clear()
        bm25_manager.clear()

        total_files = len(files)
        processed_results = []
        all_chunks, all_metadatas, all_ids = [], [], []

        for idx, file in enumerate(files, 1):
            try:
                file_name = os.path.basename(file.name)
                progress(
                    (idx - 1) / total_files,
                    desc=f"Đang xử lý tài liệu {idx}/{total_files}: {file_name}",
                )

                pages = extract_text_by_page(file.name)
                if not pages:
                    raise ValueError("Tài liệu trống hoặc không thể trích xuất văn bản")

                doc_id = f"doc_{int(time.time())}_{idx}"
                chunks = []
                metadatas = []
                for page_data in pages:
                    page_chunks = split_text(page_data["text"])
                    chunks.extend(page_chunks)
                    metadatas.extend(
                        {
                            "source": file_name,
                            "doc_id": doc_id,
                            "page": page_data["page"],
                        }
                        for _ in page_chunks
                    )
                chunk_ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]

                all_chunks.extend(chunks)
                all_metadatas.extend(metadatas)
                all_ids.extend(chunk_ids)
                page_summary = (
                    f" từ {len(pages)} trang" if pages[0]["page"] is not None else ""
                )
                processed_results.append(
                    f"{file_name}: đã tạo {len(chunks)} phân đoạn{page_summary}."
                )

            except Exception as e:
                logging.error("Không thể xử lý %s: %s", file_name, e)
                processed_results.append(f"{file_name}: xử lý thất bại – {e}")

        if all_chunks:
            progress(0.8, desc="Đang tạo embedding...")
            embeddings = encode_texts(all_chunks, show_progress=True)

            progress(0.9, desc="Đang xây chỉ mục FAISS...")
            vector_store.build_index(all_chunks, all_ids, all_metadatas, embeddings)

        progress(0.95, desc="Đang xây chỉ mục BM25...")
        bm25_manager.build_index(all_chunks, all_ids)

        summary = (
            f"\nHoàn tất: {total_files} tài liệu, {len(all_chunks)} phân đoạn."
        )
        processed_results.append(summary)
        return "\n".join(processed_results), [os.path.basename(f.name) for f in files]

    except Exception as e:
        logging.error("Quá trình xử lý tài liệu gặp lỗi: %s", e)
        return f"Không thể xử lý tài liệu: {e}", []


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
        "Model SiliconFlow": SILICONFLOW_MODEL_NAME,
        "Model OpenAI": OPENAI_MODEL_NAME,
        "Model Gemini": GEMINI_MODEL_NAME,
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
CSS = """
/* Chỉ bổ sung chi tiết, không ghi đè hành vi cốt lõi của Gradio. */
.gradio-container { max-width:100%!important; width:100%!important; }
.left-panel { padding:16px; border-radius:12px; }
.right-panel { border-radius:12px; }
.file-list { margin-top:10px; }
.footer-note { opacity:0.7; font-size:13px; margin-top:12px; }
.chunk-detail-box { min-height:200px; font-family:monospace; white-space:pre-wrap; }
.monitor-panel { border-radius:12px; padding:20px; margin-bottom:20px; }
.metric-title { font-size:14px; margin-bottom:10px; }
.metric-value { font-size:24px; font-weight:700; margin-bottom:5px; }
.metric-trend { font-size:12px; color:#4CAF50; }
.progress-container { width:100%; background:rgba(128,128,128,0.2); border-radius:10px; margin:10px 0; }
.progress-bar { height:8px; border-radius:10px;
    background:linear-gradient(90deg, #00bcd4, #7b1fa2); transition:width 0.3s ease; }
.log-container { max-height:300px; overflow-y:auto; border-radius:8px; padding:15px;
    font-family:monospace; font-size:13px; }
.theme-toggle-btn { min-width:40px!important; font-size:20px!important; padding:4px 8px!important; }
"""

# Chuyển chế độ sáng/tối bằng class ``dark`` trên phần tử body.
THEME_JS = """
(() => {
    // Khôi phục lựa chọn giao diện; mặc định là chế độ sáng.
    const saved = localStorage.getItem('rag-theme');
    if (saved === 'dark') {
        document.querySelector('body').classList.add('dark');
    }
})()
"""

def toggle_theme():
    """Để JavaScript của sự kiện nút chuyển chế độ giao diện thực thi."""
    return gr.update()

with gr.Blocks(title="learnbot_ai – Trợ lý hỏi đáp tài liệu") as demo:
    with gr.Row():
        with gr.Column(scale=9):
            gr.Markdown("# Trợ lý hỏi đáp tài liệu")
        with gr.Column(scale=2, min_width=140):
            theme_btn = gr.Button(
                "Chế độ sáng / tối",
                min_width=120,
                elem_classes="theme-toggle-btn",
            )

    with gr.Tabs() as tabs:
        # Thẻ hỏi đáp
        with gr.TabItem("Hỏi đáp"):
            with gr.Row(equal_height=True):
                with gr.Column(scale=5, elem_classes="left-panel"):
                    gr.Markdown("## Tài liệu")
                    with gr.Group():
                        gr.Markdown(
                            "Hỗ trợ PDF, Word, Excel, PowerPoint, TXT và Markdown."
                        )
                        file_input = gr.File(
                            label="Tải tài liệu",
                            file_types=[".pdf", ".txt", ".docx", ".xlsx", ".xls", ".pptx", ".md"],
                            file_count="multiple"
                        )
                        upload_btn = gr.Button("Xử lý tài liệu", variant="primary")
                        upload_status = gr.Textbox(
                            label="Trạng thái xử lý",
                            interactive=False,
                            lines=2,
                        )
                        file_list = gr.Textbox(
                            label="Tài liệu đã xử lý",
                            interactive=False,
                            lines=3,
                            elem_classes="file-list",
                        )

                    gr.Markdown("## Đặt câu hỏi")
                    with gr.Group():
                        question_input = gr.Textbox(
                            label="Câu hỏi",
                            lines=3,
                            placeholder="Ví dụ: Tài liệu nói gì về quy trình đăng ký?",
                        )
                        with gr.Row():
                            web_search_checkbox = gr.Checkbox(
                                label="Tìm thêm trên web",
                                value=False,
                                info="Cần cấu hình SERPAPI_KEY",
                            )
                            model_choice = gr.Dropdown(
                                choices=MODEL_CHOICES,
                                value=DEFAULT_MODEL_CHOICE,
                                label="Dịch vụ LLM",
                                info="Chọn dịch vụ mô hình ngôn ngữ qua API",
                            )
                        with gr.Row():
                            ask_btn = gr.Button("Gửi câu hỏi", variant="primary", scale=2)
                            clear_btn = gr.Button(
                                "Xóa cuộc trò chuyện",
                                variant="secondary",
                                elem_classes="clear-button",
                                scale=1,
                            )
                    api_info = gr.HTML("")

                with gr.Column(scale=7, elem_classes="right-panel"):
                    gr.Markdown("## Nội dung trao đổi")
                    chatbot = gr.Chatbot(label="Lịch sử trò chuyện", height=600, elem_classes="chat-container",
                                         show_label=False)
                    status_display = gr.HTML("")
                    gr.Markdown("""<div class="footer-note">
                        Câu trả lời có thể mất một đến hai phút. Bạn có thể tiếp tục hỏi dựa trên nội dung trước đó.
                    </div>""")

        # Thẻ xem phân đoạn
        with gr.TabItem("Xem phân đoạn"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("## Cấu hình hệ thống")
                    models_info = get_system_models_info()
                    with gr.Group(elem_classes="model-card"):
                        gr.Markdown("### Mô hình và kỹ thuật")
                        for key, value in models_info.items():
                            with gr.Row():
                                gr.Markdown(f"**{key}**:")
                                gr.Markdown(f"{value}")
                with gr.Column(scale=2):
                    gr.Markdown("## Thống kê phân đoạn")
                    refresh_chunks_btn = gr.Button("Tải dữ liệu phân đoạn", variant="primary")
                    chunks_status = gr.Markdown("Chọn nút trên để xem các phân đoạn đã lập chỉ mục.")
            with gr.Row():
                chunks_data = gr.Dataframe(
                    headers=["Nguồn", "Trang", "Thứ tự", "Số ký tự", "Số từ", "Nội dung xem trước"],
                    elem_classes="chunk-table",
                    interactive=False,
                    wrap=True,
                    row_count=10,
                )
            with gr.Row():
                chunk_detail_text = gr.Textbox(
                    label="Chi tiết phân đoạn",
                    placeholder="Chọn một hàng trong bảng để xem toàn bộ nội dung.",
                    lines=8, elem_classes="chunk-detail-box"
                )

        # Thẻ giám sát hệ thống
        with gr.TabItem("Giám sát hệ thống"):
            with gr.Column():
                with gr.Group(elem_classes="monitor-panel"):
                    with gr.Row():
                        gr.Markdown("## Tài nguyên hệ thống")
                        refresh_monitor_btn = gr.Button("Cập nhật số liệu", variant="primary")
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("Mức sử dụng CPU", elem_classes="metric-title")
                            cpu_value = gr.Markdown("Đang tải...", elem_classes="metric-value")
                            cpu_progress = gr.HTML('<div class="progress-container"><div class="progress-bar" style="width:0%"></div></div>')
                            cpu_info = gr.Markdown("Số lõi: đang tải...", elem_classes="metric-trend")
                        with gr.Column():
                            gr.Markdown("Mức sử dụng bộ nhớ", elem_classes="metric-title")
                            memory_value = gr.Markdown("Đang tải...", elem_classes="metric-value")
                            memory_progress = gr.HTML('<div class="progress-container"><div class="progress-bar" style="width:0%"></div></div>')
                            memory_info = gr.Markdown("Tổng bộ nhớ: đang tải...", elem_classes="metric-trend")
                        with gr.Column():
                            gr.Markdown("Dung lượng ổ đĩa", elem_classes="metric-title")
                            disk_value = gr.Markdown("Đang tải...", elem_classes="metric-value")
                            disk_progress = gr.HTML('<div class="progress-container"><div class="progress-bar" style="width:0%"></div></div>')
                            disk_info = gr.Markdown("Tổng dung lượng: đang tải...", elem_classes="metric-trend")
                        with gr.Column():
                            gr.Markdown("Kho vector", elem_classes="metric-title")
                            vector_db_value = gr.Markdown("Phân đoạn: 0", elem_classes="metric-value")
                            vector_db_info = gr.Markdown("Vector: 0", elem_classes="metric-trend")

                with gr.Group(elem_classes="monitor-panel"):
                    gr.Markdown("## Nhật ký hệ thống")
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

        api_text = """<div class="api-info" style="margin-top:10px;padding:10px;border-radius:5px;
            background:var(--panel-bg);border:1px solid var(--border-color);">
            <p><strong>Thiết lập câu hỏi</strong></p>
            <p>Tìm kiếm web: <strong>%s</strong></p>
            <p>Dịch vụ LLM: <strong>%s</strong></p>
        </div>""" % (
            "Đã bật" if enable_web_search else "Đã tắt",
            get_model_display_name(model_choice_val)
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
        return """<div class="api-info" style="margin-top:10px;padding:10px;border-radius:5px;
            background:var(--panel-bg);border:1px solid var(--border-color);">
            <p><strong>Thiết lập câu hỏi</strong></p>
            <p>Tìm kiếm web: <strong>%s</strong></p>
            <p>Dịch vụ LLM: <strong>%s</strong></p>
        </div>""" % (
            "Đã bật" if enable_web_search else "Đã tắt",
            get_model_display_name(model_choice_val)
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
            log = f'<div class="log-entry"><span style="color:var(--tech-cyan)">[{now}]</span> <span style="color:#4CAF50">[THÔNG TIN]</span> Đã cập nhật số liệu hệ thống</div>'

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
            css=CSS, js=THEME_JS
        )
    except Exception as e:
        print(f"Không thể khởi động ứng dụng: {e}")
