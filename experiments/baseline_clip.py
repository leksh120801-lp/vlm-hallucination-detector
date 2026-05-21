"""Small shim to ensure the repository root is on sys.path when this script
is run directly from inside the `experiments/` folder. This makes
`from main import main` work regardless of the current working directory.
"""
import sys
from pathlib import Path

# Add the repository root (one level above `experiments/`) to sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main import main


def run_experiment():

    print("Running baseline CLIP hallucination detection experiment\n")

    main()

if __name__ == "__main__":
    run_experiment()