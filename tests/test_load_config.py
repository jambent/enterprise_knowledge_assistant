import yaml
import pytest
from src.load_config import load_config


def test_loads_valid_yaml(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text("a: 1\nb: [1, 2]\n")
    cfg = load_config(str(p))
    assert isinstance(cfg, dict)
    assert cfg["a"] == 1
    assert cfg["b"] == [1, 2]


def test_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_config("this_file_does_not_exist.yaml")


def test_invalid_yaml_raises(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text(":\n  - bad")
    with pytest.raises(yaml.YAMLError):
        load_config(str(p))