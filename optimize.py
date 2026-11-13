"""Longer training, validation-only model selection and fresh final tests."""
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / '.packages'))
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

if __name__ == '__main__':
    from radar_control.optimization import main
    main()
