#!/bin/bash

# Define the port number
PORT=8000

# Find the process ID (PID) of the running server on the specified port
PID=$(lsof -t -i:$PORT)

# Check if the process was found and terminate it
if [ -z "$PID" ]; then
  echo "No process is running on port $PORT."
else
  echo "Stopping FastAPI server running on port $PORT with PID: $PID"
  kill -15 $PID
  echo "FastAPI server stopped."
fi
