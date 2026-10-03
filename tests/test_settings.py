import pytest

from app.settings import available_cpus


@pytest.mark.parametrize(
    ("cpu_max", "limit"),
    [("200000 100000\n", 2), ("150000 100000\n", 2), ("50000 100000\n", 1)],
)
def test_container_quota_limits_cpus(tmp_path, cpu_max, limit):
    path = tmp_path / "cpu.max"
    path.write_text(cpu_max)
    assert available_cpus(path) == min(limit, available_cpus(tmp_path / "missing"))


def test_no_quota_uses_visible_cpus(tmp_path):
    path = tmp_path / "cpu.max"
    path.write_text("max 100000\n")
    assert available_cpus(path) == available_cpus(tmp_path / "missing") >= 1
