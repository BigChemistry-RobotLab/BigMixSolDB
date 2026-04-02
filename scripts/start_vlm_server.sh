#!/bin/bash

# start_vllm_server.sh
# Wrapper script to start vLLM server
#
# Usage:
#   ./start_vlm_lighton_server.sh [additional_vllm_args...]
#
#
# Examples:
#   ./start_vlm_server.sh
#   ./start_vlm_server.sh --gpu-memory-utilization 0.8

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Shift to get additional arguments
shift || true

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}LightOn-OCR-2 vLLM Server Launcher${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Default vLLM arguments
MODEL_NAME="${MODEL_NAME:-lightonai/LightOnOCR-2-1B-bbox}"
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
API_KEY="${API_KEY:-EMPTY}"
GPU_MEMORY_UTIL="${GPU_MEMORY_UTIL:-0.9}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"

# Build vLLM command
VLLM_CMD="vllm serve \"${MODEL_NAME}\" \
    --host ${HOST} \
    --port ${PORT} \
    --api-key ${API_KEY} \
    --limit-mm-per-prompt '{\"image\": 1}' \
    --mm-processor-cache-gb 0 \
    --no-enable-prefix-caching \
    --gpu-memory-utilization ${GPU_MEMORY_UTIL} \
    --max-model-len ${MAX_MODEL_LEN}"

# Add any additional arguments passed to the script
if [ $# -gt 0 ]; then
    VLLM_CMD="${VLLM_CMD} $@"
fi

echo -e "${YELLOW}Starting vLLM server...${NC}"
echo ""
echo -e "${BLUE}Command:${NC}"
echo "$VLLM_CMD"
echo ""
echo -e "${GREEN}Server will be available at: http://${HOST}:${PORT}${NC}"
echo -e "${GREEN}API Key: ${API_KEY}${NC}"
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop the server${NC}"
echo ""

# Execute vLLM command
eval $VLLM_CMD