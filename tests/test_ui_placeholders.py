from pathlib import Path


def test_runtime_sparkline_placeholders_present():
    html = Path("central/templates/index.html").read_text(encoding="utf-8")
    assert "sparkline-util-placeholder" in html
    assert "sparkline-mem-placeholder" in html
    assert "Metrics unavailable" in html
