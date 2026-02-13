#!/bin/bash

echo "=== Atlas Backend Starting ==="
echo "Timestamp: $(date)"
echo ""

echo "Environment Variables:"
echo "  DATA_PATH: ${DATA_PATH}"
echo "  CHROMA_PATH: ${CHROMA_PATH}"
echo "  BACKEND_HOST: ${BACKEND_HOST}"
echo "  BACKEND_PORT: ${BACKEND_PORT}"
echo "  PYTHONUNBUFFERED: ${PYTHONUNBUFFERED}"
echo ""

# Verify directories exist
echo "Creating and verifying directories..."
mkdir -p "${DATA_PATH}" "${CHROMA_PATH}" 2>&1
echo "✓ Directories created"
ls -la /app/data/ 2>&1 || echo "Warning: Could not list /app/data/"
echo ""

# Check Python and dependencies
echo "Python Environment:"
python --version 2>&1
echo ""
echo "Key Dependencies:"
pip list 2>&1 | grep -E "fastapi|uvicorn|chromadb|torch|transformers|sentence-transformers" || echo "Warning: Some packages not found"
echo ""

# Test imports before starting
echo "Testing Python imports..."
python -c "
import sys
print('Python executable:', sys.executable)
print('Python path:', sys.path)
print()

try:
    import fastapi
    print('✓ fastapi imported')
except Exception as e:
    print('✗ fastapi import failed:', str(e))
    sys.exit(1)

try:
    import uvicorn
    print('✓ uvicorn imported')
except Exception as e:
    print('✗ uvicorn import failed:', str(e))
    sys.exit(1)

try:
    import chromadb
    print('✓ chromadb imported')
except Exception as e:
    print('✗ chromadb import failed:', str(e))
    sys.exit(1)

try:
    import torch
    print('✓ torch imported')
except Exception as e:
    print('✗ torch import failed:', str(e))
    sys.exit(1)

try:
    import sentence_transformers
    print('✓ sentence_transformers imported')
except Exception as e:
    print('✗ sentence_transformers import failed:', str(e))
    sys.exit(1)

print()
print('All imports successful!')
" 2>&1

if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: Import test failed!"
    echo "Container will exit now. Check the logs above for details."
    exit 1
fi

echo ""
echo "Starting FastAPI application..."
echo "Command: python main.py"
echo ""
echo "⏳ IMPORTANT: First startup downloads ML model (~400MB), may take 2-3 minutes"
echo "⏳ Please wait... watching for progress..."
echo ""

# Start the application and capture any errors
python main.py 2>&1 || {
    EXIT_CODE=$?
    echo ""
    echo "ERROR: Application crashed with exit code: $EXIT_CODE"
    echo "Timestamp: $(date)"
    exit $EXIT_CODE
}
