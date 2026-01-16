#!/bin/bash
# Start both the API server and frontend dev server

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "Stopping servers..."
    kill $API_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    exit
}

trap cleanup INT TERM

# Check if API server dependencies are installed
echo "Checking Python dependencies..."
if ! python -c "import fastapi, uvicorn, polars" 2>/dev/null; then
    echo "⚠ Missing Python dependencies."
    echo "Installing dependencies with uv sync..."
    if command -v uv &> /dev/null; then
        uv sync
    else
        echo "Error: 'uv' command not found. Please install uv or run: uv sync"
        exit 1
    fi
    echo "✓ Dependencies installed"
fi

# Start API server in background
echo "Starting API server on port 8000..."
uv run python api_server.py > api_server.log 2>&1 &
API_PID=$!

# Wait for API server to be ready
echo "Waiting for API server to start..."
for i in {1..15}; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "✓ API server is ready!"
        break
    fi
    if [ $i -eq 15 ]; then
        echo "Error: API server failed to start after 15 seconds."
        echo "Check api_server.log for details:"
        tail -20 api_server.log
        cleanup
        exit 1
    fi
    sleep 1
    echo -n "."
done
echo ""

# Check if frontend dependencies are installed
if [ ! -d "frontend/node_modules" ]; then
    echo "Frontend dependencies not found. Installing..."
    cd frontend
    npm install
    cd ..
fi

# Start frontend dev server
echo "Starting frontend dev server on port 3000..."
cd frontend
npm run dev > ../frontend.log 2>&1 &
FRONTEND_PID=$!
cd ..

echo ""
echo "✓ Both servers started!"
echo "  API: http://localhost:8000"
echo "  Frontend: http://localhost:3000"
echo ""
echo "Logs:"
echo "  API: tail -f api_server.log"
echo "  Frontend: tail -f frontend.log"
echo ""
echo "Press Ctrl+C to stop both servers"

# Wait for processes
wait

