#!/bin/bash

# Face Verification System - Development Startup Script
echo "🚀 Starting Face Verification System..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check if required tools are installed
echo -e "${BLUE}📋 Checking prerequisites...${NC}"

if ! command_exists node; then
    echo -e "${RED}❌ Node.js is not installed${NC}"
    echo "Please install Node.js 16+ from https://nodejs.org"
    exit 1
fi

if ! command_exists npm; then
    echo -e "${RED}❌ npm is not installed${NC}"
    exit 1
fi

if ! command_exists python3; then
    echo -e "${RED}❌ Python 3 is not installed${NC}"
    echo "Please install Python 3.8+ from https://python.org"
    exit 1
fi

echo -e "${GREEN}✅ All prerequisites found${NC}"

# Get the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Start backend server
echo -e "${BLUE}🔧 Starting backend server...${NC}"
cd "$PROJECT_ROOT/server"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}📦 Creating Python virtual environment...${NC}"
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install Python dependencies
if [ -f "requirements.txt" ]; then
    echo -e "${YELLOW}📦 Installing Python dependencies...${NC}"
    pip install -r requirements.txt
fi

# Start backend server in background
echo -e "${GREEN}🚀 Starting FastAPI server on port 8000...${NC}"
python main.py &
BACKEND_PID=$!

# Wait for backend to start
sleep 3

# Check if backend is running
if ! curl -f http://localhost:8000/docs >/dev/null 2>&1; then
    echo -e "${YELLOW}⏳ Waiting for backend to start...${NC}"
    sleep 5
fi

# Start frontend development server
echo -e "${BLUE}🎨 Starting frontend development server...${NC}"
cd "$PROJECT_ROOT/web"

# Install npm dependencies if node_modules doesn't exist
if [ ! -d "node_modules" ]; then
    echo -e "${YELLOW}📦 Installing npm dependencies...${NC}"
    npm install
fi

# Start frontend server
echo -e "${GREEN}🚀 Starting React development server on port 5173...${NC}"
npm run dev &
FRONTEND_PID=$!

# Function to cleanup processes on exit
cleanup() {
    echo -e "\n${YELLOW}🛑 Shutting down servers...${NC}"
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    echo -e "${GREEN}✅ Cleanup completed${NC}"
}

# Set trap to cleanup on script exit
trap cleanup EXIT INT TERM

# Wait for a moment then show status
sleep 3

echo -e "\n${GREEN}🎉 Face Verification System is running!${NC}"
echo -e "${BLUE}📱 Frontend:${NC} http://localhost:5173"
echo -e "${BLUE}🔧 Backend API:${NC} http://localhost:8000"
echo -e "${BLUE}📖 API Documentation:${NC} http://localhost:8000/docs"
echo -e "\n${YELLOW}Press Ctrl+C to stop all servers${NC}\n"

# Keep script running
wait