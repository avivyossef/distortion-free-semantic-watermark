import sys
from pathlib import Path

# Tests import ``watermark`` and ``encoder`` from the public tree root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
