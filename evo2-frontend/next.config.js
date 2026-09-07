/**
 * Run `build` or `dev` with `SKIP_ENV_VALIDATION` to skip env validation. This is especially useful
 * for Docker builds.
 */
import "./src/env.js";

/** @type {import("next").NextConfig} */
const config = {
  reactStrictMode: false,

  // The scaffolded code carries ~140 pre-existing lint errors (mostly `any`
  // from untyped NCBI/UCSC JSON responses). `next build` fails on lint errors
  // by default, which blocks deploys for style issues. Linting is still run
  // deliberately via `npm run lint` / `npm run check`.
  eslint: {
    ignoreDuringBuilds: true,
  },
};

export default config;
