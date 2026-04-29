#!/bin/bash
# Kill Qwen3.6-35B-A3B-NVFP4 serving stack

# Kill by process name
pkill -f "qwen3_6_server.py" 2>/dev/null
pkill -f "server_compress.py" 2>/dev/null
pkill -f "qwen_litellm.py" 2>/dev/null
pkill -f "qwen_token_stats_server" 2>/dev/null

# Kill orphaned vLLM engine cores (can survive pkill by name)
pkill -9 -f "VLLM::EngineCore" 2>/dev/null

# Kill any processes on our ports using lsof (macOS-compatible)
for port in 11114 11115 11116 11124 11125 11126 11134 11135 11136; do
    pids=$(lsof -ti :$port 2>/dev/null)
    if [ -n "$pids" ]; then
        echo "Killing processes on port $port: $pids"
        echo "$pids" | xargs kill -9 2>/dev/null
    fi
done

echo "✅ Qwen serving processes killed"

# Verify ports are free
if [ "$(uname)" = "Darwin" ]; then
    # macOS: use lsof
    for port in 11114 11115 11116 11124 11125 11126 11134 11135 11136; do
        if lsof -i :$port -t >/dev/null 2>&1; then
            echo "⚠️  Port $port still in use"
        else
            echo "  Port $port is free"
        fi
    done
else
    # Linux: use ss
    ss -tlnp 2>/dev/null | grep -E '11114|11115|11116|11124|11125|11126|11134|11135|11136' || echo "  (ports are free)"
fi

ps aux | grep -E 'qwen3_6|server_compress|qwen_token|qwen_litellm' | grep -v grep | grep -v pkill || echo "  (no remaining processes)"
