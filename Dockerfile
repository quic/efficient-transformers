FROM docker-registry.qualcomm.com/qraniumtest/qranium:1.23.0.41-ubuntu22-x86_64

RUN apt-get update && apt-get install -y \
    git \
    tmux \
    vim \
    python3.12 \
    python3.12-venv \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . /app/qefficient-library

RUN python3.12 -m venv /app/llm_env

ENV PATH="/app/llm_env/bin:$PATH"

RUN pip install --upgrade pip

WORKDIR /app/qefficient-library

RUN python3.12 -m pip install .
RUN pip install torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu

# Entrypoint
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh

RUN chmod +x /usr/local/bin/docker-entrypoint.sh

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]

CMD ["sleep", "infinity"]
