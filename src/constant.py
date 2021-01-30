# coding: utf-8
from os import makedirs
from os.path import abspath, dirname
from pathlib import Path

PROJECT_ROOT = Path(dirname(abspath(__file__))).parent
DATA_DIR = PROJECT_ROOT / 'data'
MODEL_DIR = PROJECT_ROOT / 'model'

makedirs(MODEL_DIR, exist_ok=True)
