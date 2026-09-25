# thoth-node dashboard

React 19 + TypeScript + Vite app served by the node on **:80**
(plans/CONTRACT.md §5). Tabs: Status · Live (3D room) · Captures ·
Models · Automations · Metadata/Room editor.

## layout

- `src/scene/` — **mirror of `website/src/scene/`** (the shared
  room/v1 visualization). It is a straight file copy; edit upstream in
  the website repo or keep them in lockstep.
- `src/api.ts` — API client. Local mode hits `/api/*` with the node's
  `local_token` (`?token=` once → `thoth_dash` cookie + localStorage).
  Portal embed: `?relay=<brain>/<device-id>` rewrites every call to
  `POST {relay}/api`; `?viewer=portal` hides the token field and
  local-only links (capture zip download).
- `src/pages/` — one tab per page.

## build

```bash
npm install
npm run build        # tsc -b && vite build → ../thoth/dashboard/dist
```

The `dist/` output is committed (nodes install via `pip git+…` — no npm
on the Pi) and packaged via `tool.setuptools.package-data` +
`MANIFEST.in`, so `python -m thoth daemon` serves it on :80.

## dev

```bash
VITE_THOTH_API=http://10.0.0.88:5001 npm run dev
```

proxies `/api/*` to a live node (add `?token=<local_token>` to the URL
once to unlock).
