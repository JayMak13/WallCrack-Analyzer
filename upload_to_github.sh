#!/bin/bash
# ============================================================
# Wall Crack Analyzer - GitHub Upload Script
# Target: https://github.com/JayMak13/WallCrack-Analyzer
# ============================================================

REPO_URL="https://github.com/JayMak13/WallCrack-Analyzer.git"

echo "Wall Crack Analyzer - GitHub Upload Helper"
echo "=========================================="
echo ""
echo "Target Repository: $REPO_URL"
echo ""

# Check if git is installed
if ! command -v git &> /dev/null; then
    echo "ERROR: git is not installed. Please install git first."
    echo "  Ubuntu/Debian: sudo apt-get install git"
    echo "  macOS: brew install git"
    echo "  Windows: download from git-scm.com"
    exit 1
fi

echo "Step 1: Cloning your repository..."
git clone $REPO_URL wallcrack-temp
cd wallcrack-temp

echo ""
echo "Step 2: Copying files..."
cp ../app.py .
cp ../requirements.txt .
cp ../README.md .

echo ""
echo "Step 3: Adding files to git..."
git add app.py requirements.txt README.md

echo ""
echo "Step 4: Committing..."
git commit -m "Add Wall Crack Analyzer app with real structural engineering data

Features:
- Image analysis using Sobel edge detection and PCA orientation analysis
- 8 crack types with real identification criteria from structural engineering sources
- Detailed repair methods with actual cost estimates
- Streamlit web interface for easy deployment
- Text and JSON report generation

Data sources: The Basement Guide, JLB Foundation, Mainmark, AquaGuard Injection,
Dalinghaus Construction, AZTA Engineering, ARDEX ENDURA"

echo ""
echo "Step 5: Pushing to GitHub..."
git push origin main

echo ""
echo "=========================================="
echo "DONE! Files uploaded successfully."
echo ""
echo "Next steps:"
echo "  1. Go to https://share.streamlit.io"
echo "  2. Sign in with GitHub"
echo "  3. Click 'New app'"
echo "  4. Select JayMak13/WallCrack-Analyzer"
echo "  5. Main file path: app.py"
echo "  6. Click Deploy!"
echo ""

# Cleanup
cd ..
rm -rf wallcrack-temp
