#!/bin/bash
# Double-click in Finder to start the Image Crawler GUI (macOS).
# The first start installs everything, which takes a few minutes.
cd "$(dirname "$0")" || exit 1
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/opt/homebrew/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
  echo "uv is not installed. Install it with:"
  echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
  read -r -p "Press Enter to close."
  exit 1
fi
uv run --managed-python image-crawler gui
