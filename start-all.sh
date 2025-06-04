#!/usr/bin/env sh

uvicorn app.main:app --host 0.0.0.0 --port 8000 &
echo "🚀 uvicorn backend started on http://0.0.0.0:8000"

cd app/frontend
npm run start --hostname 0.0.0.0 -- -p 3000 &
echo "🚀 Next.js frontend started on http://0.0.0.0:3000"

cd /app
node proxy-server.js
