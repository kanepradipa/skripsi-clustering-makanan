#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Food Clustering - Starting Services${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

echo "Activating virtual environment..."
source venv/bin/activate

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Starting Backend (Flask) on port 5000${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Start backend in background
cd backend
python app.py &
BACKEND_PID=$!
echo -e "${GREEN}Backend started with PID: $BACKEND_PID${NC}"

# Wait for backend to start
sleep 3

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Starting Frontend (Streamlit) on port 8501${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Start frontend
cd ../frontend
streamlit run app.py

# Cleanup on exit
echo ""
echo "Shutting down services..."
kill $BACKEND_PID
echo -e "${GREEN}All services stopped${NC}"
