import type { NextConfig } from "next";

const config: NextConfig = {
  output: "export",
  assetPrefix: ".",
  outputFileTracingRoot: process.cwd(),
  allowedDevOrigins: ["127.0.0.1", "localhost"],
  images: { unoptimized: true },
  trailingSlash: true,
};

export default config;
