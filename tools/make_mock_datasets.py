"""Create FAKE SisFall-format and KFall-format datasets from the simulator, to rehearse the real-data commands.

    python tools/make_mock_datasets.py
    python scripts/run_real_data.py --dataset sisfall --path data/mock/sisfall
    python scripts/run_real_data.py --dataset kfall   --path data/mock/kfall

Needs: pip install openpyxl   (KFall labels are Excel files). Results on this data mean nothing about real falls.
"""
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from prefall.mockdata import write_mock_kfall, write_mock_sisfall

meta = write_mock_sisfall("data/mock/sisfall", n_subjects=10)
print(f"wrote {len(meta)} fake SisFall files to data/mock/sisfall")
meta = write_mock_kfall("data/mock/kfall", n_subjects=10)
print(f"wrote {len(meta)} fake KFall files to data/mock/kfall (+ label_data/)")
