const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL?.replace(/\/$/, "") ??
  "http://127.0.0.1:17995";

export type Provider = "siliconflow" | "openai" | "gemini";

export type AnswerStatus =
  | "answered"
  | "insufficient_evidence"
  | "empty_knowledge_base"
  | "error";

export type Citation = {
  type: string;
  source?: string;
  page?: number;
  url?: string;
};

export type StructuredCitation = {
  document: string;
  page: number | null;
  chunk_id: string;
  score: number | null;
  type: "document" | "web";
  url: string | null;
};

export type SystemStatus = {
  status: string;
  siliconflow_configured: boolean;
  openai_configured: boolean;
  gemini_configured: boolean;
  llm_provider: Provider;
  serpapi_configured: boolean;
  vector_store_ready: boolean;
  total_chunks: number;
  index_snapshot_id?: string | null;
  min_relevance_score: number;
  version: string;
};

export type UploadResult = {
  status: "success" | "error";
  message: string;
  file_info?: {
    filename: string;
    chunks: number;
    processed_files: number;
    skipped_files: number;
  };
};

export type StoredDocument = {
  id: string;
  source_name: string;
  file_size: number;
  status: "processing" | "ready" | "failed";
  chunk_count: number;
  created_at: string;
  updated_at: string;
};

export type AnswerResult = {
  answer: string;
  answer_status: AnswerStatus;
  citations: StructuredCitation[];
  sources: Citation[];
  metadata: {
    enable_web_search: boolean;
    model: string;
    citation_count: number;
    min_relevance_score: number;
  };
};

export class ApiError extends Error {
  constructor(message: string, public readonly status?: number) {
    super(message);
    this.name = "ApiError";
  }
}

async function parseResponse<T>(response: Response): Promise<T> {
  if (response.ok) return (await response.json()) as T;

  let message = "Không thể kết nối với máy chủ.";
  try {
    const payload = (await response.json()) as { detail?: string; message?: string };
    message = payload.detail ?? payload.message ?? message;
  } catch {
    if (response.statusText) message = response.statusText;
  }
  throw new ApiError(message, response.status);
}

export async function getSystemStatus(signal?: AbortSignal) {
  const response = await fetch(`${API_BASE_URL}/api/status`, {
    signal,
    cache: "no-store",
  });
  return parseResponse<SystemStatus>(response);
}

export async function getDocuments(signal?: AbortSignal) {
  const response = await fetch(`${API_BASE_URL}/api/documents`, {
    signal,
    cache: "no-store",
  });
  return parseResponse<StoredDocument[]>(response);
}

export async function uploadDocument(file: File) {
  const body = new FormData();
  body.append("file", file);
  const response = await fetch(`${API_BASE_URL}/api/upload`, {
    method: "POST",
    body,
  });
  return parseResponse<UploadResult>(response);
}

export async function askQuestion(
  question: string,
  enableWebSearch: boolean,
  modelChoice: Provider,
  signal?: AbortSignal,
) {
  const response = await fetch(`${API_BASE_URL}/api/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      question,
      enable_web_search: enableWebSearch,
      model_choice: modelChoice,
    }),
    signal,
  });
  return parseResponse<AnswerResult>(response);
}

export function getProviderLabel(provider: Provider) {
  return {
    siliconflow: "SiliconFlow",
    openai: "OpenAI",
    gemini: "Gemini",
  }[provider];
}
