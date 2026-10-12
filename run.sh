#!/usr/bin/env bash
# Evidence-MAD: Start both backend and frontend
set -e

cd "$(dirname "$0")"
source /home/ssp/projects/Python/myenv/bin/activate

# Kill existing servers on ports 8000 and 3000
kill $(lsof -t -i:8000) 2>/dev/null || true
kill $(lsof -t -i:3000) 2>/dev/null || true
sleep 1

# Start backend
uvicorn api:app --host 127.0.0.1 --port 8000 &
BACKEND_PID=$!

# Start frontend
python -m http.server 3000 --bind 127.0.0.1 --directory frontend &
FRONTEND_PID=$!

echo ""
echo "========================================"
echo "  Evidence-MAD is running!"
echo "========================================"
echo "  Frontend:  http://127.0.0.1:3000"
echo "  Backend:   http://127.0.0.1:8000"
echo "  API Docs:  http://127.0.0.1:8000/docs"
echo "========================================"
echo ""
echo "Press Ctrl+C to stop both servers."
echo ""

# Wait for either to exit
wait $BACKEND_PID $FRONTEND_PID
