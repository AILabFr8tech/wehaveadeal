#!/bin/sh
# If your agents need extra packages, list them in agents/requirements.txt.
set -e
if [ -f /app/agents/requirements.txt ]; then
  echo "Installing agents/requirements.txt ..."
  pip install --no-cache-dir -r /app/agents/requirements.txt
fi
exec wehaveadeal serve --host 0.0.0.0 --port 8000 --agents-dir /app/agents
