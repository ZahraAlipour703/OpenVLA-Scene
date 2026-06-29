from pathlib import Path

from omegaconf import OmegaConf


def load_config(path: str):

    path = Path(path)

    if not path.exists():

        raise FileNotFoundError(path)

    return OmegaConf.load(path)