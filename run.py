"""Command line entry point. Local SUMO packages are isolated in .packages."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / '.packages'))
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

if __name__ == '__main__':
    from radar_control.experiment import main
    main()
