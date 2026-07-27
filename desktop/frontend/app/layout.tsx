import type { Metadata } from "next";
import type { ReactNode } from "react";

import "./globals.css";

export const metadata: Metadata = {
  title: "FineSub Desktop",
  description: "专注、可靠的本地字幕工作台",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="zh-CN">
      <head>
        <link rel="icon" href="./icon.png" type="image/png" />
      </head>
      <body>{children}</body>
    </html>
  );
}
