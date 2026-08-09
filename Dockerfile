FROM docker-registry.qualcomm.com/library/ubuntu:24.04

RUN apt-get update && apt-get install -y \
    git \
    tmux \
    vim \
    python3.12 \
    python3.12-venv \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# pip recognizes this variable
ENV PIP_CACHE_DIR=/var/cache/pip

WORKDIR /app

RUN mkdir -p /app/qefficient-library
COPY . /app/qefficient-library

# Create Virtual Env for the docker image
RUN python3.12 -m venv /app/llm_env
ENV PATH="/app/llm_env/bin:$PATH"

RUN pip install --upgrade pip

WORKDIR /app/qefficient-library
RUN python3.12 -m pip install .
RUN pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu

CMD ["sleep", "infinity"]
