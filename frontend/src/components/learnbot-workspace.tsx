"use client";

import {
  type ChangeEvent,
  type DragEvent,
  type FormEvent,
  type KeyboardEvent,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  ApiError,
  askQuestion,
  type AnswerStatus,
  deleteDocument,
  getDocuments,
  getProviderLabel,
  getSystemStatus,
  type Provider,
  type StructuredCitation,
  type SystemStatus,
  uploadDocument,
} from "@/lib/api";
import { CitationList } from "./citation-list";
import { Icon } from "./icons";

type DocumentState = {
  id: string;
  name: string;
  chunks?: number;
  state: "processing" | "ready" | "error";
  message?: string;
  persisted: boolean;
};

type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
  answerStatus?: AnswerStatus;
  citations?: StructuredCitation[];
};

const suggestions = [
  "Tóm tắt các luận điểm chính trong tài liệu",
  "Liệt kê những số liệu quan trọng và nguồn tương ứng",
  "So sánh các quan điểm được đề cập trong tài liệu",
];

function newId() {
  return globalThis.crypto?.randomUUID?.() ?? `${Date.now()}-${Math.random()}`;
}

function readableError(error: unknown) {
  if (error instanceof ApiError) return error.message;
  if (error instanceof Error && error.name === "AbortError") return "Yêu cầu đã được hủy.";
  return "Không thể kết nối với backend. Hãy kiểm tra FastAPI đang chạy.";
}

export function LearnBotWorkspace() {
  const [theme, setTheme] = useState<"light" | "dark">("light");
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [status, setStatus] = useState<SystemStatus | null>(null);
  const [statusError, setStatusError] = useState("");
  const [documents, setDocuments] = useState<DocumentState[]>([]);
  const [documentsLoaded, setDocumentsLoaded] = useState(false);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [question, setQuestion] = useState("");
  const [provider, setProvider] = useState<Provider>("siliconflow");
  const [webSearch, setWebSearch] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [deletingDocumentId, setDeletingDocumentId] = useState<string | null>(null);
  const [isAsking, setIsAsking] = useState(false);
  const [uploadError, setUploadError] = useState("");
  const [composerError, setComposerError] = useState("");
  const fileInputRef = useRef<HTMLInputElement>(null);
  const messageScrollRef = useRef<HTMLElement>(null);
  const formRef = useRef<HTMLFormElement>(null);

  const refreshStatus = useCallback(async () => {
    try {
      setStatusError("");
      const nextStatus = await getSystemStatus();
      setStatus(nextStatus);
      setProvider(nextStatus.llm_provider);
    } catch (error) {
      setStatusError(readableError(error));
    }
  }, []);

  const refreshDocuments = useCallback(async () => {
    try {
      const storedDocuments = await getDocuments();
      setDocuments(
        storedDocuments.map((document) => ({
          id: document.id,
          name: document.source_name,
          chunks: document.chunk_count,
          state: document.status === "failed" ? "error" : document.status,
          persisted: true,
        })),
      );
    } catch (error) {
      setUploadError(readableError(error));
    } finally {
      setDocumentsLoaded(true);
    }
  }, []);

  useEffect(() => {
    const savedTheme = localStorage.getItem("learnbot-theme");
    const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
    setTheme(savedTheme === "dark" || (!savedTheme && prefersDark) ? "dark" : "light");
    void refreshStatus();
    void refreshDocuments();
  }, [refreshDocuments, refreshStatus]);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem("learnbot-theme", theme);
  }, [theme]);

  useEffect(() => {
    const messageScroll = messageScrollRef.current;
    if (!messageScroll) return;

    const animationFrame = window.requestAnimationFrame(() => {
      messageScroll.scrollTo({
        top: messageScroll.scrollHeight,
        behavior: "smooth",
      });
    });

    return () => window.cancelAnimationFrame(animationFrame);
  }, [messages, isAsking]);

  const latestCitations = useMemo(
    () =>
      [...messages].reverse().find((message) => message.role === "assistant")
        ?.citations ?? [],
    [messages],
  );

  const processFiles = useCallback(
    async (files: File[]) => {
      if (!files.length || isUploading) return;
      setUploadError("");
      setIsUploading(true);

      const queued = files.map<DocumentState>((file) => ({
        id: newId(),
        name: file.name,
        state: "processing",
        persisted: false,
      }));
      setDocuments((current) => [...queued, ...current]);

      for (let index = 0; index < files.length; index += 1) {
        const file = files[index];
        const documentId = queued[index].id;
        try {
          const result = await uploadDocument(file);
          if (result.status === "error") setUploadError(result.message);
          setDocuments((current) =>
            current.map((document) =>
              document.id === documentId
                ? {
                    ...document,
                    state: result.status === "success" ? "ready" : "error",
                    chunks: result.file_info?.chunks,
                    message: result.message,
                  }
                : document,
            ),
          );
        } catch (error) {
          const message = readableError(error);
          setUploadError(message);
          setDocuments((current) =>
            current.map((document) =>
              document.id === documentId
                ? { ...document, state: "error", message }
                : document,
            ),
          );
        }
      }

      await refreshDocuments();
      setIsUploading(false);
      setSidebarOpen(false);
      await refreshStatus();
    },
    [isUploading, refreshDocuments, refreshStatus],
  );

  const handleDeleteDocument = useCallback(
    async (document: DocumentState) => {
      if (
        !document.persisted ||
        deletingDocumentId ||
        !window.confirm(
          `Xóa “${document.name}” khỏi kho tri thức? Hành động này không thể hoàn tác.`,
        )
      ) {
        return;
      }

      setUploadError("");
      setDeletingDocumentId(document.id);
      try {
        await deleteDocument(document.id);
        setDocuments((current) =>
          current.filter((item) => item.id !== document.id),
        );
        await refreshStatus();
      } catch (error) {
        setUploadError(readableError(error));
      } finally {
        setDeletingDocumentId(null);
      }
    },
    [deletingDocumentId, refreshStatus],
  );

  const handleFiles = (event: ChangeEvent<HTMLInputElement>) => {
    void processFiles(Array.from(event.target.files ?? []));
    event.target.value = "";
  };

  const handleDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    void processFiles(Array.from(event.dataTransfer.files));
  };

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const prompt = question.trim();
    if (!prompt || isAsking) return;

    setComposerError("");
    setQuestion("");
    setMessages((current) => [
      ...current,
      { id: newId(), role: "user", content: prompt },
    ]);
    setIsAsking(true);

    try {
      const result = await askQuestion(prompt, webSearch, provider);
      setMessages((current) => [
        ...current,
        {
          id: newId(),
          role: "assistant",
          content: result.answer,
          answerStatus: result.answer_status,
          citations: result.citations,
        },
      ]);
    } catch (error) {
      setComposerError(readableError(error));
      setQuestion(prompt);
    } finally {
      setIsAsking(false);
    }
  };

  const handleComposerKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
      event.preventDefault();
      formRef.current?.requestSubmit();
    }
  };

  return (
    <>
      <a className="skip-link" href="#chat-main">Chuyển đến hội thoại</a>
      <div className="app-shell">
        <button
          className={`drawer-backdrop ${sidebarOpen ? "is-visible" : ""}`}
          aria-label="Đóng danh sách tài liệu"
          onClick={() => setSidebarOpen(false)}
        />

        <aside className={`document-sidebar ${sidebarOpen ? "is-open" : ""}`} aria-label="Tài liệu">
          <div className="brand-row">
            <span className="brand-mark"><Icon name="book" /></span>
            <div><strong>LearnBot</strong><span>Không gian tài liệu</span></div>
            <button className="icon-button drawer-close" onClick={() => setSidebarOpen(false)} aria-label="Đóng panel tài liệu"><Icon name="x" /></button>
          </div>

          <div className="sidebar-heading">
            <div><p className="eyebrow">Kho làm việc</p><h2>Tài liệu</h2></div>
            <span className="document-count">{documents.filter((item) => item.state === "ready").length}</span>
          </div>

          <div
            className={`upload-zone ${isUploading ? "is-busy" : ""}`}
            onDragOver={(event) => event.preventDefault()}
            onDrop={handleDrop}
          >
            <Icon name="upload" />
            <strong>{isUploading ? "Đang lập chỉ mục" : "Thêm tài liệu"}</strong>
            <span>PDF, DOCX, XLSX, PPTX, TXT hoặc Markdown</span>
            <button type="button" onClick={() => fileInputRef.current?.click()} disabled={isUploading}>Chọn tệp</button>
            <input ref={fileInputRef} type="file" multiple hidden accept=".pdf,.txt,.docx,.xlsx,.xls,.pptx,.md" onChange={handleFiles} />
          </div>

          {uploadError && <p className="inline-error" role="alert">{uploadError}</p>}

          <div className="document-list" aria-live="polite">
            {!documentsLoaded ? (
              <div className="quiet-state"><Icon name="file" /><p>Đang tải tài liệu…</p></div>
            ) : documents.length === 0 ? (
              <div className="quiet-state"><Icon name="file" /><p>Chưa có tài liệu</p><span>Tệp đã xử lý sẽ xuất hiện tại đây.</span></div>
            ) : documents.map((document) => (
              <div className="document-row" key={document.id}>
                <span className={`file-state ${document.state}`}><Icon name={document.state === "ready" ? "check" : "file"} /></span>
                <div><strong title={document.name}>{document.name}</strong><span>{deletingDocumentId === document.id ? "Đang xóa…" : document.state === "processing" ? "Đang xử lý…" : document.state === "error" ? "Xử lý thất bại" : `${document.chunks ?? 0} phân đoạn`}</span></div>
                {document.persisted && (
                  <button
                    className="document-delete"
                    type="button"
                    aria-label={`Xóa tài liệu ${document.name}`}
                    disabled={isUploading || deletingDocumentId !== null}
                    onClick={() => void handleDeleteDocument(document)}
                  >
                    <Icon name="trash" />
                  </button>
                )}
              </div>
            ))}
          </div>

          <div className="sidebar-status">
            <span className={`status-dot ${status?.status === "healthy" ? "online" : ""}`} />
            <div><strong>{status?.status === "healthy" ? "Backend sẵn sàng" : "Chưa kết nối backend"}</strong><span>{status ? `Phiên bản ${status.version}` : statusError || "Đang kiểm tra…"}</span></div>
            <button className="icon-button" onClick={() => { void refreshStatus(); void refreshDocuments(); }} aria-label="Kiểm tra lại kết nối"><Icon name="refresh" /></button>
          </div>
        </aside>

        <main className="chat-main" id="chat-main">
          <header className="chat-header">
            <div className="chat-title-row">
              <button className="icon-button mobile-menu" onClick={() => setSidebarOpen(true)} aria-label="Mở danh sách tài liệu"><Icon name="menu" /></button>
              <div><p className="eyebrow">Hỏi đáp tài liệu</p><h1>Cuộc trò chuyện mới</h1></div>
            </div>
            <div className="header-actions">
              <span className="context-summary"><Icon name="database" />{status?.total_chunks ?? 0} phân đoạn</span>
              <button className="theme-button" onClick={() => setTheme(theme === "light" ? "dark" : "light")} aria-label="Đổi chế độ màu"><Icon name={theme === "light" ? "moon" : "sun"} /><span>{theme === "light" ? "Tối" : "Sáng"}</span></button>
            </div>
          </header>

          <section ref={messageScrollRef} className="message-scroll" aria-label="Nội dung hội thoại">
            <div className="message-column" aria-live="polite">
              {messages.length === 0 ? (
                <div className="conversation-empty">
                  <span className="empty-symbol"><Icon name="book" /></span>
                  <p className="eyebrow">Bắt đầu từ tài liệu của bạn</p>
                  <h2>Đặt câu hỏi, nhận câu trả lời có nguồn.</h2>
                  <p>LearnBot truy xuất nội dung liên quan trước khi gọi LLM, giúp bạn kiểm tra lại thông tin theo tài liệu gốc.</p>
                  <div className="suggestion-list">
                    {suggestions.map((suggestion) => <button key={suggestion} onClick={() => setQuestion(suggestion)}>{suggestion}<Icon name="chevron" /></button>)}
                  </div>
                </div>
              ) : messages.map((message) => (
                <article
                  className={`message ${message.role} ${message.answerStatus ? `status-${message.answerStatus}` : ""}`}
                  key={message.id}
                >
                  <div className="message-author">{message.role === "user" ? "Bạn" : "LearnBot"}</div>
                  {message.role === "assistant" && message.answerStatus && message.answerStatus !== "answered" && (
                    <div
                      className="answer-status"
                      role={message.answerStatus === "error" ? "alert" : "status"}
                    >
                      <Icon name={message.answerStatus === "error" ? "x" : "database"} />
                      <strong>
                        {message.answerStatus === "insufficient_evidence"
                          ? "Không đủ bằng chứng"
                          : message.answerStatus === "empty_knowledge_base"
                            ? "Kho tri thức đang trống"
                            : "Không thể tạo câu trả lời"}
                      </strong>
                    </div>
                  )}
                  <div className={`message-content ${message.role === "assistant" ? "markdown-body" : ""}`}>
                    {message.role === "assistant" ? (
                      <ReactMarkdown remarkPlugins={[remarkGfm]}>
                        {message.answerStatus && message.answerStatus !== "answered"
                          ? message.content.replace(/^⚠️\s*/u, "")
                          : message.content}
                      </ReactMarkdown>
                    ) : message.content}
                  </div>
                  {!!message.citations?.length && (
                    <section className="message-citations" aria-label="Nguồn đối chiếu cho câu trả lời">
                      <div className="citation-heading">
                        <h3 id={`citations-${message.id}`}>Nguồn đối chiếu</h3>
                        <span>{message.citations.length} nguồn</span>
                      </div>
                      <CitationList
                        citations={message.citations}
                        labelledBy={`citations-${message.id}`}
                      />
                    </section>
                  )}
                </article>
              ))}
              {isAsking && <div className="thinking" role="status"><span /><span /><span /><b>Đang đọc tài liệu và soạn câu trả lời</b></div>}
            </div>
          </section>

          <div className="composer-wrap">
            <form className="composer" ref={formRef} onSubmit={handleSubmit}>
              <label htmlFor="question">Câu hỏi</label>
              <textarea id="question" value={question} onChange={(event) => setQuestion(event.target.value)} onKeyDown={handleComposerKeyDown} placeholder="Hỏi về nội dung, số liệu hoặc luận điểm trong tài liệu…" rows={2} disabled={isAsking} />
              {composerError && <div className="composer-error" role="alert"><span>{composerError}</span><button type="button" onClick={() => setComposerError("")}>Đóng</button></div>}
              <div className="composer-toolbar">
                <div className="answer-options">
                  <label className="switch-row"><input type="checkbox" checked={webSearch} onChange={(event) => setWebSearch(event.target.checked)} /><span><Icon name="globe" />Web</span></label>
                  <label className="provider-select"><span className="sr-only">Dịch vụ LLM</span><select value={provider} onChange={(event) => setProvider(event.target.value as Provider)}><option value="siliconflow">SiliconFlow</option><option value="openai">OpenAI</option><option value="gemini">Gemini</option></select></label>
                </div>
                <div className="composer-actions">
                  {messages.length > 0 && <button className="clear-button" type="button" onClick={() => setMessages([])} aria-label="Xóa hội thoại"><Icon name="trash" /></button>}
                  <button className="send-button" type="submit" disabled={!question.trim() || isAsking}><span>{isAsking ? "Đang trả lời" : "Gửi"}</span><Icon name="send" /></button>
                </div>
              </div>
            </form>
            <p className="composer-hint">Enter để gửi · Shift + Enter để xuống dòng · Kiểm tra nguồn trước khi sử dụng thông tin quan trọng.</p>
          </div>
        </main>

        <aside className="context-panel" aria-label="Nguồn và trạng thái">
          <div className="context-section">
            <p className="eyebrow">Cấu hình</p><h2>Phiên làm việc</h2>
            <dl className="detail-list"><div><dt>Dịch vụ</dt><dd>{getProviderLabel(provider)}</dd></div><div><dt>Tìm kiếm web</dt><dd>{webSearch ? "Đang bật" : "Đang tắt"}</dd></div><div><dt>Kho vector</dt><dd>{status?.vector_store_ready ? "Sẵn sàng" : "Chưa có dữ liệu"}</dd></div></dl>
          </div>
          <div className="context-section sources-section">
            <div className="section-title"><div><p className="eyebrow">Đối chiếu</p><h2 id="latest-citations-title">Nguồn gần nhất</h2></div><span>{latestCitations.length}</span></div>
            {latestCitations.length === 0 ? <div className="quiet-state compact"><Icon name="file" /><p>Chưa có nguồn</p><span>Nguồn của câu trả lời gần nhất sẽ xuất hiện tại đây.</span></div> : <CitationList citations={latestCitations} compact labelledBy="latest-citations-title" />}
          </div>
          <div className="context-note"><strong>Giới hạn</strong><p>Câu trả lời có thể thiếu ngữ cảnh. Luôn mở lại tài liệu gốc khi cần độ chính xác cao.</p></div>
        </aside>
      </div>
    </>
  );
}
