# Create T3 App

This is a [T3 Stack](https://create.t3.gg/) project bootstrapped with `create-t3-app`.

## What's next? How do I make an app with this?

We try to keep this project as simple as possible, so you can start with just the scaffolding we set up for you, and add additional things later when they become necessary.

If you are not familiar with the different technologies used in this project, please refer to the respective docs. If you still are in the wind, please join our [Discord](https://t3.gg/discord) and ask for help.

- [Next.js](https://nextjs.org)
- [NextAuth.js](https://next-auth.js.org)
- [Prisma](https://prisma.io)
- [Drizzle](https://orm.drizzle.team)
- [Tailwind CSS](https://tailwindcss.com)
- [tRPC](https://trpc.io)

## Learn More

To learn more about the [T3 Stack](https://create.t3.gg/), take a look at the following resources:

- [Documentation](https://create.t3.gg/)
- [Learn the T3 Stack](https://create.t3.gg/en/faq#what-learning-resources-are-currently-available) — Check out these awesome tutorials

You can check out the [create-t3-app GitHub repository](https://github.com/t3-oss/create-t3-app) — your feedback and contributions are welcome!

## Local development

```bash
npm install
cp .env.example .env   # then fill in the Modal endpoint URL
npm run dev
```

`NEXT_PUBLIC_ANALYZE_SINGLE_VARIANT_BASE_URL` is the web URL printed by
`modal deploy evo2-backend/main.py` for `Evo2Model.analyze_single_variant`.
The app talks to it directly from the browser; Modal reflects the request
origin back, so no CORS configuration is needed.

## Deploying to Vercel

This repo is a monorepo and the Next.js app is **not** at the repo root, so a
default Vercel import builds nothing and every route returns
`404: NOT_FOUND`. Two project settings are required:

1. **Settings -> Build & Deployment -> Root Directory**: set to `evo2-frontend`.
2. **Settings -> Environment Variables**: add
   `NEXT_PUBLIC_ANALYZE_SINGLE_VARIANT_BASE_URL` for Production, Preview and
   Development. `src/env.js` validates it at build time, so the build fails
   without it.

Settings changes do not retrigger a build — redeploy the latest deployment
afterwards.

Note that `NEXT_PUBLIC_*` values are inlined into the client bundle, so the
Modal endpoint is publicly callable from any deployed page. It runs on an
H100, so put a token or a Next.js route-handler proxy in front of it before
sharing the site widely.
