"""Package source, simulated data, trained models and reports, excluding runtimes."""
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent
TARGET = ROOT / '作业成果_代码数据与报告.zip'
INCLUDE = ['run.py', 'requirements.txt', 'README.md', '运行作业.ps1', '作业要求与数据方案.md',
           '数据说明.md', 'verify_results.py', 'verify_optimized.py', 'optimize.py',
           'package_results.py', '毫米波雷达轨迹感知自适应信号控制课程报告_DQN与PPO.docx', 'radar_control', 'tests', 'results/full', 'results/optimized']


def main():
    count = 0
    with zipfile.ZipFile(TARGET, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name in INCLUDE:
            source = ROOT / name
            paths = sorted(source.rglob('*')) if source.is_dir() else [source]
            for path in paths:
                if not path.is_file() or '__pycache__' in path.parts or path.suffix == '.pyc':
                    continue
                relative_parts = path.relative_to(ROOT).parts
                # Per-episode raw training logs remain on disk. The archive keeps
                # demand data, history, checkpoints, and all validation/test evidence.
                if 'training' in relative_parts and any(p.startswith('episode_') for p in relative_parts):
                    continue
                archive.write(path, path.relative_to(ROOT).as_posix())
                count += 1
    with zipfile.ZipFile(TARGET) as archive:
        assert archive.testzip() is None
        for name in ['results/optimized/优化训练报告.html',
                     'results/optimized/comparison/figures/00_final_demand.png',
                     'results/optimized/comparison/figures/07_final_signal.png',
                     'radar_control/optimization_report.py']:
            assert archive.read(name) == (ROOT / name).read_bytes()
    print(f'Packaged {count} files; {TARGET.stat().st_size / 1024 ** 2:.2f} MiB')


if __name__ == '__main__':
    main()
