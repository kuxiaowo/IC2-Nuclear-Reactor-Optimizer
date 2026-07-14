import os

from ic2_reactor.launcher import frontend_build_required


def test_frontend_build_is_required_when_output_is_missing(tmp_path):
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "page.tsx").write_text("export default 1", encoding="utf-8")
    assert frontend_build_required(tmp_path)


def test_frontend_build_detects_sources_newer_than_output(tmp_path):
    source = tmp_path / "app" / "page.tsx"
    output = tmp_path / "dist" / "index.html"
    source.parent.mkdir()
    output.parent.mkdir()
    source.write_text("export default 1", encoding="utf-8")
    output.write_text("built", encoding="utf-8")

    os.utime(source, ns=(1_000_000_000, 1_000_000_000))
    os.utime(output, ns=(2_000_000_000, 2_000_000_000))
    assert not frontend_build_required(tmp_path)

    os.utime(source, ns=(3_000_000_000, 3_000_000_000))
    assert frontend_build_required(tmp_path)
