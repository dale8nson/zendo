#!/usr/bin/env sh

# Start the FastAPI backend
uvicorn app.main:app --host 0.0.0.0 --port 8000 &
# echo "🚀 FastAPI backend started on http://0.0.0.0:8000"

# Serve the prebuilt frontend using a static file server like `serve` or `http-server`
# If using `serve`, install it during build: npm install -g serve


# If you're using something like nginx instead, then nginx must be the entrypoint

# Start backend
# cd /app/backend/python
# uvicorn app.main:app --host 0.0.0.0 --port 8000 &
# echo "🚀 uvicorn backend started on http://0.0.0.0:8000"

# # Start frontend
# cd /app/frontend
# if ! command -v npm > /dev/null; then
#   echo "❌ npm not found in final image. Skipping frontend startup."
#   exit 1
# fi

# npm run build
# npm run start &
# echo "🚀 Next.js frontend started on http://0.0.0.0:3000"

# # Optional: start proxy (if defined in package.json)
# if npm run | grep -q proxy; then
#   npm run proxy
# else
#   echo "⚠️  'proxy' script not found in package.json"
# fi
#
