import type { NextConfig } from "next";
import { userInfo } from "node:os";

const isCodexSandbox = userInfo().username.toLowerCase().includes("codexsandbox");

const nextConfig: NextConfig = {
  reactStrictMode: true,
  agentRules: false,
  // Codex và người dùng Windows chạy bằng các tài khoản khác nhau.
  // Tách thư mục build để tránh file trong .next bị trộn quyền sở hữu.
  distDir: isCodexSandbox ? ".next-codex" : ".next-local",
};

export default nextConfig;
