#!/bin/bash
# Start LiteLLM proxy to route MiniMax-M2.7 → local Qwen (port 11112)
# Usage: ./start_minimax_proxy.sh

cd /home/sna/ai-projects/lunch-model

PORT=11115
CONFIG=lite_llm_config_minimax.yaml

echo "Starting LiteLLM proxy on port $PORT..."
echo "  config: $CONFIG"
echo "  backend: localhost:11112 (Qwen3.6-35B-A3B-NVFP4)"
echo ""

./venv/bin/litellm \
    --config "$CONFIG" \
    --port "$PORT" \
    --host 0.0.0.0 \
    --alias "MiniMax-M2.7 → Qwen3.6-35B-NVFP4"