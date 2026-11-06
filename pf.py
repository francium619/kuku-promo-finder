"""Launcher that works from any folder: `python pf.py run ...` == `python -m promo_finder run ...`."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from promo_finder.cli import main  # noqa: E402

sys.exit(main())
