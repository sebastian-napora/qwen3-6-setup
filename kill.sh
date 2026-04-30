#!/bin/bash
# Kill Qwen3.6-35B-A3B-NVFP4 serving stack.
# Local RAG / embeddings run inside the LiteLLM proxy on port 11111.

# Kill by process name
pkill -f "qwen3_6_server.py" 2>/dev/null
pkill -f "server_compress.py" 2>/dev/null
pkill -f "qwen-compress" 2>/dev/null
pkill -f "qwen_local_rag" 2>/dev/null
pkill -f "mlx-start.sh" 2>/dev/null
pkill -f "qwen_litellm.py" 2>/dev/null
pkill -f "qwen_token_stats_server" 2>/dev/null

# Kill orphaned vLLM engine cores (can survive pkill by name)
pkill -9 -f "VLLM::EngineCore" 2>/dev/null

# Kill any server on our ports (orphaned or missed)
fuser -k 11111/tcp 2>/dev/null
fuser -k 11112/tcp 2>/dev/null
fuser -k 11113/tcp 2>/dev/null

# Final sweep: kill any lingering process on our ports.
for port in 11111 11112 11113; do
    pid=$(ss -tlnp 2>/dev/null | grep ":$port " | grep -oP 'pid=\K[0-9]+')
    if [ -n "$pid" ]; then
        kill -9 "$pid" 2>/dev/null
        echo "Killed PID $pid on port $port"
    fi
    if command -v lsof >/dev/null 2>&1; then
        for lsof_pid in $(lsof -ti tcp:"$port" 2>/dev/null); do
            kill -9 "$lsof_pid" 2>/dev/null
            echo "Killed PID $lsof_pid on port $port"
        done
    fi
done

echo "✅ Qwen serving processes killed"
ss -tlnp 2>/dev/null | grep -E '11111|11112|11113' || echo "  (ports are free)"
ps aux | grep -E 'qwen|QWEN_RAG|VLLM|server_compress|mlx-start' | grep -v grep | grep -v pkill || echo "  (no remaining processes)"
