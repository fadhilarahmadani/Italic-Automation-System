#!/bin/bash

# KBBI Database Setup Script
# Downloads the KBBI dictionary database for dual verification

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Project root directory (assuming script is in scripts/)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${PROJECT_ROOT}/src/data"
KBBI_FILE="${DATA_DIR}/dictionary_JSON.json"
KBBI_URL="https://raw.githubusercontent.com/dyazincahya/KBBI-SQL-database/master/dictionary_JSON.json"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  KBBI Database Setup${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check if curl is installed
if ! command -v curl &> /dev/null; then
    echo -e "${RED}Error: curl is not installed${NC}"
    echo "Please install curl and try again"
    exit 1
fi

# Create data directory if it doesn't exist
if [ ! -d "$DATA_DIR" ]; then
    echo -e "${YELLOW}Creating data directory...${NC}"
    mkdir -p "$DATA_DIR"
fi

# Check if file already exists
if [ -f "$KBBI_FILE" ]; then
    echo -e "${YELLOW}KBBI database already exists at:${NC}"
    echo "  $KBBI_FILE"
    echo ""
    read -p "Do you want to re-download it? (y/N): " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${GREEN}Keeping existing file. Setup complete!${NC}"
        exit 0
    fi
    echo -e "${YELLOW}Removing existing file...${NC}"
    rm "$KBBI_FILE"
fi

# Download KBBI database
echo -e "${BLUE}Downloading KBBI database...${NC}"
echo "  Source: ${KBBI_URL}"
echo "  Target: ${KBBI_FILE}"
echo ""

if curl -L -o "$KBBI_FILE" "$KBBI_URL" --progress-bar; then
    echo ""
    echo -e "${GREEN}✓ Download complete!${NC}"

    # Verify file
    if [ -f "$KBBI_FILE" ]; then
        FILE_SIZE=$(du -h "$KBBI_FILE" | cut -f1)
        echo ""
        echo -e "${GREEN}✓ File verified${NC}"
        echo "  Size: ${FILE_SIZE}"

        # Check if it's valid JSON (just first few bytes)
        if head -c 100 "$KBBI_FILE" | grep -q "{"; then
            echo -e "${GREEN}✓ JSON format verified${NC}"
        else
            echo -e "${RED}⚠ Warning: File may not be valid JSON${NC}"
        fi

        echo ""
        echo -e "${GREEN}========================================${NC}"
        echo -e "${GREEN}  KBBI Database Setup Complete!${NC}"
        echo -e "${GREEN}========================================${NC}"
        echo ""
        echo "The KBBI database is ready to use."
        echo "Start your FastAPI server to load it:"
        echo ""
        echo "  cd ${PROJECT_ROOT}"
        echo "  python -m uvicorn src.api.main:app --reload"
        echo ""
    else
        echo -e "${RED}✗ Error: File not found after download${NC}"
        exit 1
    fi
else
    echo ""
    echo -e "${RED}✗ Download failed${NC}"
    echo ""
    echo "Please try downloading manually:"
    echo "  1. Visit: https://github.com/dyazincahya/KBBI-SQL-database"
    echo "  2. Download dictionary_JSON.json"
    echo "  3. Place it in: ${DATA_DIR}/"
    exit 1
fi
