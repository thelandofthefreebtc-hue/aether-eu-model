#!/bin/bash
cd ~/Downloads
source aether_venv/bin/activate

echo "🛑 Killing any old AetherEU server..."
lsof -ti:8000 | xargs kill -9 2>/dev/null || true

echo "🚀 Starting AetherEU Sovereign AI v2.3 (No Torch)..."
python aether_eu_model_no_torch.py
