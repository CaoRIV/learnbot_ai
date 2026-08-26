import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "LearnBot — Hỏi đáp tài liệu tiếng Việt",
  description: "Trợ lý hỏi đáp tài liệu tiếng Việt dựa trên truy xuất kết hợp và LLM qua API.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="vi" suppressHydrationWarning>
      <body>{children}</body>
    </html>
  );
}
