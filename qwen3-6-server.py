#!/usr/bin/env python3
"""
Local vLLM API Server for Qwen3.6-35B-A3B-NVFP4 on NVIDIA GB10
Uses vLLM's built-in API server with:
  --tool-call-parser qwen3_xml   (parses XML tool call format)
  --enable-auto-tool-choice       (auto-selects tools when needed)
"""

import sys
import os

# Allow long max_model_len (model's config says 40960 but NVFP4 supports 131072)
os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"

# Ensure venv packages take priority
_venv_bin = os.path.join(os.path.dirname(__file__), "venv", "bin")
if _venv_bin not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _venv_bin + os.pathsep + os.environ.get("PATH", "")

# Add venv to Python path so 'vllm' resolves
_venv_lib = os.path.join(os.path.dirname(__file__), "venv", "lib", "python3.12", "site-packages")
if _venv_lib not in sys.path:
    sys.path.insert(0, _venv_lib)

from vllm.entrypoints.openai.api_server import run_server
from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args


def main():
    import argparse

    # Build the parser the same way CLI does
    from vllm.utils.argparse_utils import FlexibleArgumentParser
    parser = FlexibleArgumentParser(prog="lunch")
    subparsers = parser.add_subparsers(dest="command")
    serve_parser = subparsers.add_parser(
        "serve",
        help="Launch vLLM server",
        usage="lunch serve [model_tag] [options]",
    )

    # Wire up all the vLLM serve arguments
    from vllm.entrypoints.openai.cli_args import make_arg_parser
    serve_parser = make_arg_parser(serve_parser)

    # Build CLI args -- positional model_tag goes directly to serve subparser
    argv = [
        "RedHatAI/Qwen3.6-35B-A3B-NVFP4",
        # Model loading
        "--trust-remote-code",
        "--dtype", "bfloat16",
        "--max-model-len", "131072",
        "--gpu-memory-utilization", "0.90",
        # MoE backend for NVFP4
        "--moe-backend", "cutlass",
        # Performance / compatibility
        "--enforce-eager",
        "--disable-log-stats",
        # Tool calling -- THIS IS THE KEY ADDITION
        "--tool-call-parser", "qwen3_xml",
        "--enable-auto-tool-choice",
        # API server
        "--port", "1111",
        "--host", "0.0.0.0",
    ]

    # Parse directly against serve_parser (not parent parser)
    args = serve_parser.parse_args(argv)
    # Tag on the serve subcommand for the CLI handler
    args.command = "serve"
    args.model_tag = argv[0]  # RedHatAI/Qwen3.6-35B-A3B-NVFP4

    # Apply the same model_tag→model mapping the CLI handler does
    args.model = args.model_tag

    validate_parsed_serve_args(args)

    print(f"\n🚀 Blackwell NVFP4 Server @ 128K Context")
    print(f"📡 API: http://0.0.0.0:1111/v1")
    print(f"🔧 Tool parser: qwen3_xml (XML format)")
    print(f"🧠 Thinking blocks: VISIBLE (when tools provided)")
    print(f"🔧 Tool calling: ENABLED")
    print(f"\n   Tool parser info:")
    print(f"   - Extracts <tool_call><function=...><parameter=...> XML format")
    print(f"   - Converts to OpenAI tool_calls format automatically")
    print(f"   - Handles streaming tool calls correctly")
    print()

    import uvloop
    uvloop.run(run_server(args))


if __name__ == "__main__":
    main()
