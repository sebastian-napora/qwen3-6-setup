python3 -m pip install --upgrade pip

pip install vllm \
  --index-url https://wheels.vllm.ai/0.19.1/cu130 \
  --extra-index-url https://download.pytorch.org/whl/cu130 \
  --extra-index-url https://pypi.org/simple \
  --force-reinstall
