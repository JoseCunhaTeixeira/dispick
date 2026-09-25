# dispick training image: CUDA PyTorch, the synthesis, and the pipeline script.
#
#   docker build -f docker/Dockerfile -t dispick-train .
#   docker run --gpus all -v /data/dispick:/work dispick-train
#
# The host needs an NVIDIA driver (>= 560 for CUDA 12.6) and the NVIDIA container toolkit;
# the CUDA libraries come with PyTorch's wheels. See docs/cloud.md.
FROM ghcr.io/astral-sh/uv:python3.14-bookworm-slim

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONUNBUFFERED=1

WORKDIR /opt/dispick

# Dependencies first, for the layer cache: they change far less than the code.
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project --extra synth --extra train --extra cuda --extra onnx

COPY src ./src
COPY configs ./configs
COPY scripts ./scripts
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra synth --extra train --extra cuda --extra onnx

ENV PATH="/opt/dispick/.venv/bin:${PATH}" \
    DISPICK_WORK=/work

# SageMaker starts the image with the argument "train"; the script ignores it.
ENTRYPOINT ["/opt/dispick/scripts/cloud_train.sh"]
