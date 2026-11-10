# Backend for the local dashboard: the arena, the nine reference architectures
# and every agent in ./agents (mounted at runtime -- edit files, no rebuild needed).
FROM python:3.11-slim

WORKDIR /app
# CPU-only torch keeps the image small; the RL baselines only need inference.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml README.md ./
COPY wehaveadeal ./wehaveadeal
RUN pip install --no-cache-dir ".[server,llm]"

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8000
ENTRYPOINT ["/entrypoint.sh"]
