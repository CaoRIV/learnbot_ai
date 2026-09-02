import type { StructuredCitation } from "@/lib/api";
import { Icon } from "./icons";

type CitationListProps = {
  citations: StructuredCitation[];
  compact?: boolean;
  labelledBy?: string;
};

const INLINE_CITATION_LIMIT = 4;
const COMPACT_CITATION_LIMIT = 5;

function safeExternalUrl(url: string | null) {
  if (!url) return null;
  try {
    const parsed = new URL(url);
    return parsed.protocol === "http:" || parsed.protocol === "https:"
      ? parsed.toString()
      : null;
  } catch {
    return null;
  }
}

function formatScore(score: number | null) {
  if (score === null || !Number.isFinite(score)) return null;
  const normalizedScore = Math.max(0, Math.min(1, score));
  return `${Math.round(normalizedScore * 100)}% liên quan`;
}

function CitationRow({ citation }: { citation: StructuredCitation }) {
  const externalUrl = safeExternalUrl(citation.url);
  const score = formatScore(citation.score);
  const sourceKind = citation.type === "web" ? "Nguồn web" : "Tài liệu";
  const metadata = [
    citation.page !== null ? `Trang ${citation.page}` : sourceKind,
    score,
  ].filter(Boolean);
  const content = (
    <>
      <span className="citation-icon">
        <Icon name={citation.type === "web" ? "globe" : "file"} />
      </span>
      <span className="citation-copy">
        <strong title={citation.document}>{citation.document}</strong>
        <small>{metadata.join(" / ")}</small>
        <code title={citation.chunk_id}>{citation.chunk_id}</code>
      </span>
      {externalUrl && <Icon name="chevron" />}
    </>
  );

  return externalUrl ? (
    <a
      className="citation-row"
      href={externalUrl}
      target="_blank"
      rel="noreferrer"
      aria-label={`Mở nguồn web ${citation.document}`}
    >
      {content}
    </a>
  ) : (
    <div className="citation-row is-static">{content}</div>
  );
}

export function CitationList({ citations, compact = false, labelledBy }: CitationListProps) {
  const visibleLimit = compact ? COMPACT_CITATION_LIMIT : INLINE_CITATION_LIMIT;
  const visible = citations.slice(0, visibleLimit);
  const remaining = citations.slice(visibleLimit);

  return (
    <ul
      className={`citation-list ${compact ? "is-compact" : ""}`}
      aria-labelledby={labelledBy}
    >
      {visible.map((citation) => (
        <li key={citation.chunk_id}>
          <CitationRow citation={citation} />
        </li>
      ))}
      {remaining.length > 0 && (
        <li className="citation-more">
          <details>
            <summary>Xem thêm {remaining.length} nguồn</summary>
            <ul>
              {remaining.map((citation) => (
                <li key={citation.chunk_id}>
                  <CitationRow citation={citation} />
                </li>
              ))}
            </ul>
          </details>
        </li>
      )}
    </ul>
  );
}
