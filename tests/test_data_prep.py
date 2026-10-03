import pytest
import yaml

import download


def test_polygon_becomes_bounding_box():
    assert download.polygon_line_to_box("0 0.1 0.2 0.5 0.2 0.5 0.6") == "0 0.300000 0.400000 0.400000 0.400000"


def test_box_line_is_unchanged():
    assert download.polygon_line_to_box("1 0.5 0.5 0.2 0.2") == "1 0.5 0.5 0.2 0.2"


@pytest.fixture
def dataset(tmp_path, monkeypatch):
    monkeypatch.setattr(download, "DATASET_DIR", tmp_path)
    labels = tmp_path / "train" / "labels"
    labels.mkdir(parents=True)
    (labels / "a.txt").write_text("0 0.5 0.5 0.9 0.9\n1 0.4 0.4 0.1 0.1\n")
    (tmp_path / "data.yaml").write_text(yaml.safe_dump({"names": ["Diamond", "Inclusion"]}))
    return tmp_path


def test_filter_classes_drops_and_renumbers(dataset):
    kept = download.filter_classes(download.read_names(), ["inclusion"])

    assert kept == ["Inclusion"]
    assert (dataset / "train" / "labels" / "a.txt").read_text() == "0 0.4 0.4 0.1 0.1\n"


def test_filter_classes_all_keeps_everything(dataset):
    assert download.filter_classes(download.read_names(), ["all"]) == ["Diamond", "Inclusion"]


def test_filter_classes_rejects_unknown(dataset):
    with pytest.raises(SystemExit):
        download.filter_classes(download.read_names(), ["Ruby"])
