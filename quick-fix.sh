#!/bin/bash

# Quick Fix Script for ME2 Branch
# Author: Claude Code Assistant
# Date: 2026-01-22

set -e  # Exit on error

echo "=================================="
echo "🔧 QUICK FIX SCRIPT - ME2 BRANCH"
echo "=================================="
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Step 1: Check Python
echo -e "${YELLOW}[1/6]${NC} Checking Python..."
if ! command -v python &> /dev/null; then
    echo -e "${RED}❌ Python not found!${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Python found: $(python --version)${NC}"
echo ""

# Step 2: Install Python dependencies
echo -e "${YELLOW}[2/6]${NC} Installing Python dependencies..."
echo "This may take 5-10 minutes..."
pip install -q -r requirements.txt
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Python dependencies installed${NC}"
else
    echo -e "${RED}❌ Failed to install dependencies${NC}"
    echo "Try manually: pip install -r requirements.txt"
    exit 1
fi
echo ""

# Step 3: Check if model exists
echo -e "${YELLOW}[3/6]${NC} Checking ML model..."
if [ -d "models/indobert-italic/final" ]; then
    echo -e "${GREEN}✅ ML model found${NC}"
else
    echo -e "${RED}❌ ML model not found!${NC}"
    echo "You need to train the model first:"
    echo "  python src/train.py"
    exit 1
fi
echo ""

# Step 4: Test API
echo -e "${YELLOW}[4/6]${NC} Testing API..."
echo "Starting API in background..."
nohup python src/api/main.py > api.log 2>&1 &
API_PID=$!
echo "API PID: $API_PID"

# Wait for API to start
echo "Waiting for API to start..."
for i in {1..10}; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo -e "${GREEN}✅ API is running${NC}"
        break
    fi
    sleep 1
    if [ $i -eq 10 ]; then
        echo -e "${RED}❌ API failed to start${NC}"
        echo "Check api.log for errors:"
        tail -20 api.log
        exit 1
    fi
done
echo ""

# Step 5: Rebuild Word Add-in
echo -e "${YELLOW}[5/6]${NC} Rebuilding Word Add-in..."
cd word-addin
rm -rf dist node_modules/.cache
npm run build:dev > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Word Add-in rebuilt${NC}"
else
    echo -e "${RED}❌ Failed to build Word Add-in${NC}"
    exit 1
fi
cd ..
echo ""

# Step 6: Summary
echo "=================================="
echo -e "${GREEN}✅ ALL CHECKS PASSED!${NC}"
echo "=================================="
echo ""
echo "Next steps:"
echo "1. Start Word Add-in dev server:"
echo "   cd word-addin && npm run dev-server"
echo ""
echo "2. In another terminal, sideload add-in to Word:"
echo "   cd word-addin && npm run start"
echo ""
echo "3. Clear Word cache if needed:"
echo "   - Windows: Delete %LOCALAPPDATA%\\Microsoft\\Office\\16.0\\Wef\\"
echo "   - macOS: rm -rf ~/Library/Containers/com.microsoft.Word/Data/Library/Caches/"
echo ""
echo "API is running in background (PID: $API_PID)"
echo "To stop API: kill $API_PID"
echo "To view API logs: tail -f api.log"
echo ""
echo -e "${GREEN}Happy coding! 🚀${NC}"
