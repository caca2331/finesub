import type { NextConfig } from "next";

const isProd = process.env.NODE_ENV === "production";

const config: NextConfig = {
  ...(isProd
    ? {
        output: "export",
        assetPrefix: ".",
        trailingSlash: true,
      }
    : {}),
  outputFileTracingRoot: process.cwd(),
  allowedDevOrigins: ["127.0.0.1", "localhost"],
  images: { unoptimized: true },
};

export default config;
