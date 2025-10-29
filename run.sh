#!/bin/bash

# Start backend
echo "Starting Flask backend..."
cd backend
python app.py &
BACKEND_PID=$!

# Wait for backend to start
sleep 2

# Start frontend
echo "Starting Streamlit frontend..."
cd ../frontend
streamlit run app.py

# Cleanup
kill $BACKEND_PID
