# Refund Sentinel Frontend

React + TypeScript investigation console for Refund Sentinel.

## Development

```bash
npm ci
npm run lint
npm run build
npm run dev
```

The development server listens on port `5000` and proxies `/api` and `/health` to the backend at `http://127.0.0.1:8000`.

Create `.env` from `.env.example` and set `VITE_API_KEY` to the same value as the backend `APP_API_KEY`.

Do not commit `.env`, `node_modules`, or `dist`.
