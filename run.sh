#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON_BIN=""
if command -v python3.12 >/dev/null 2>&1; then
  PYTHON_BIN="python3.12"
elif command -v python3.11 >/dev/null 2>&1; then
  PYTHON_BIN="python3.11"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
else
  echo "Python 3.11+ is required. Install it from https://www.python.org/downloads/"
  exit 1
fi

if [ ! -d ".venv" ]; then
  echo "Creating virtual environment with ${PYTHON_BIN}..."
  "$PYTHON_BIN" -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

python -m pip install --quiet --upgrade pip
python -m pip install --quiet -r requirements.txt

find_ollama() {
  if [ -n "${OLLAMA_BIN:-}" ] && [ -x "${OLLAMA_BIN}" ]; then
    echo "${OLLAMA_BIN}"
    return 0
  fi
  if command -v ollama >/dev/null 2>&1; then
    command -v ollama
    return 0
  fi
  for candidate in /opt/homebrew/bin/ollama /usr/local/bin/ollama "$HOME/.ollama/bin/ollama"; do
    if [ -x "$candidate" ]; then
      echo "$candidate"
      return 0
    fi
  done
  return 1
}

OLLAMA_BIN="$(find_ollama || true)"
if [ -z "$OLLAMA_BIN" ]; then
  echo "Ollama was not found. Install it first:"
  echo "  brew install ollama"
  echo "Then run this script again."
  exit 1
fi

if ! curl -sf "http://127.0.0.1:11434/api/version" >/dev/null 2>&1; then
  echo "Starting Ollama server..."
  OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0 \
    nohup "$OLLAMA_BIN" serve > "${TMPDIR:-/tmp}/narrator_ollama.log" 2>&1 &
  for _ in $(seq 1 30); do
    curl -sf "http://127.0.0.1:11434/api/version" >/dev/null 2>&1 && break
    sleep 1
  done
fi

VISION_MODEL="${VISION_MODEL:-gemma3:4b}"
NARRATOR_MODEL="${NARRATOR_MODEL:-gemma3:4b}"

for model in "$VISION_MODEL" "$NARRATOR_MODEL"; do
  if ! "$OLLAMA_BIN" show "$model" >/dev/null 2>&1; then
    echo "Downloading model: ${model} (one-time, requires internet)..."
    "$OLLAMA_BIN" pull "$model"
  fi
done

KOKORO_DIR="${KOKORO_MODEL_DIR:-models/tts}"
KOKORO_ONNX="${KOKORO_DIR}/kokoro-v1.0.onnx"
KOKORO_VOICES="${KOKORO_DIR}/voices-v1.0.bin"
KOKORO_BASE="https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"

download_kokoro() {
  mkdir -p "$KOKORO_DIR"
  echo "Downloading Kokoro neural voice model (one-time, ~350 MB)..."
  curl -fSL --retry 3 -C - -o "$KOKORO_ONNX" "${KOKORO_BASE}/kokoro-v1.0.onnx" || return 1
  curl -fSL --retry 3 -C - -o "$KOKORO_VOICES" "${KOKORO_BASE}/voices-v1.0.bin" || return 1
}

if [ ! -f "$KOKORO_ONNX" ] || [ ! -f "$KOKORO_VOICES" ]; then
  download_kokoro || echo "WARNING: Kokoro download failed; the app will fall back to macOS 'say'."
fi

PORT="${PORT:-8765}"
URL="http://127.0.0.1:${PORT}"

if command -v open >/dev/null 2>&1; then
  ( sleep 2; open "$URL" >/dev/null 2>&1 || true ) &
fi

echo ""
echo "AI Life Documentary is starting at ${URL}"
echo "Vision model:   ${VISION_MODEL}"
echo "Narrator model: ${NARRATOR_MODEL}"
echo "Press Ctrl+C to stop."
echo ""

exec python -m uvicorn backend.main:app --host 127.0.0.1 --port "$PORT"
