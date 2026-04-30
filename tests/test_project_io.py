"""Tests for msnoise.core.project_io and msnoise.project.

Unit tests cover:
- scan_lineages: glob pattern reconstruction
- build_params_from_project_yaml: layer extraction + meta-key stripping
- _pair_from_stem: pair format conversion
- LEVEL_GLOBS / LEVEL_CATEGORIES consistency

Integration tests cover:
- export_project → extract_archive → MSNoiseProject.from_project_dir → list()
- params.yaml round-trip (written on first list(), reused on second)
- Multi-lineage project (two filter sets)
"""
import datetime
import json
import tarfile
import tempfile
from pathlib import Path

import pytest
import yaml

from msnoise.core.project_io import (
    LEVEL_CATEGORIES,
    LEVEL_GLOBS,
    _pair_from_stem,
    build_params_from_project_yaml,
    export_project,
    extract_archive,
    file_sha256,
    scan_lineages,
)
from msnoise.project import MSNoiseProject


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_project(tmp_path):
    """Minimal project tree with stack + refstack outputs for two filter sets."""
    project_yaml = {
        "msnoise_project_version": 1,
        "global_1": {
            "startdate": "2014-01-01",
            "enddate": "2014-12-31",
            "output_folder": str(tmp_path),
        },
        "preprocess_1": {"after": "global_1", "cc_sampling_rate": 50.0},
        "cc_1": {"after": "preprocess_1", "corr_duration": 3600.0},
        "filter_1": {"after": "cc_1", "freqmin": 0.1, "freqmax": 1.0},
        "filter_2": {"after": "cc_1", "freqmin": 1.0, "freqmax": 5.0},
        "stack_1": {"after": "filter_1", "stack_method": "linear"},
        "stack_2": {"after": "filter_2", "stack_method": "linear"},
        "refstack_1": {"after": "filter_1", "stack_method": "linear"},
        "refstack_2": {"after": "filter_2", "stack_method": "linear"},
    }
    (tmp_path / "project.yaml").write_text(yaml.dump(project_yaml))

    for filt in ("filter_1", "filter_2"):
        filt_n = filt.split("_")[1]
        for cat in (f"stack_{filt_n}", f"refstack_{filt_n}"):
            out = (
                tmp_path
                / "global_1" / "preprocess_1" / "cc_1"
                / filt / cat / "_output"
            )
            out.mkdir(parents=True)
            (out / "BE.UCC..HHZ_BE.MEM..HHZ.nc").write_bytes(b"fake netcdf")

    return tmp_path


# ---------------------------------------------------------------------------
# Unit tests — _pair_from_stem
# ---------------------------------------------------------------------------

class TestPairFromStem:
    def test_simple(self):
        assert _pair_from_stem("BE.UCC..HHZ_BE.MEM..HHZ") == "BE.UCC..HHZ:BE.MEM..HHZ"

    def test_only_first_underscore(self):
        # Station names contain dots only; the separator is the single _
        result = _pair_from_stem("NET.STA.00.HHZ_NET.STA.10.HHZ")
        assert result == "NET.STA.00.HHZ:NET.STA.10.HHZ"

    def test_autocorrelation(self):
        # Same station both sides
        assert _pair_from_stem("BE.UCC..HHZ_BE.UCC..HHZ") == "BE.UCC..HHZ:BE.UCC..HHZ"


# ---------------------------------------------------------------------------
# Unit tests — scan_lineages
# ---------------------------------------------------------------------------

class TestScanLineages:
    def test_finds_stack_dirs(self, simple_project):
        results = scan_lineages(simple_project, "stack")
        assert len(results) == 2
        names = [r[0] for r in results]
        assert ["global_1", "preprocess_1", "cc_1", "filter_1", "stack_1"] in names
        assert ["global_1", "preprocess_1", "cc_1", "filter_2", "stack_2"] in names

    def test_sorted_output(self, simple_project):
        results = scan_lineages(simple_project, "stack")
        names = [r[0] for r in results]
        assert names == sorted(names)

    def test_refstack_dirs(self, simple_project):
        results = scan_lineages(simple_project, "refstack")
        assert len(results) == 2

    def test_missing_output_dir_excluded(self, tmp_path):
        """Dirs without _output/ child must not appear."""
        project_yaml = {"msnoise_project_version": 1}
        (tmp_path / "project.yaml").write_text(yaml.dump(project_yaml))
        # Create stack_1 dir but no _output inside
        (tmp_path / "global_1" / "stack_1").mkdir(parents=True)
        results = scan_lineages(tmp_path, "stack")
        assert results == []

    def test_empty_project(self, tmp_path):
        (tmp_path / "project.yaml").write_text(yaml.dump({}))
        assert scan_lineages(tmp_path, "stack") == []


# ---------------------------------------------------------------------------
# Unit tests — build_params_from_project_yaml
# ---------------------------------------------------------------------------

class TestBuildParams:
    def test_layers_created(self, simple_project):
        yaml_path = simple_project / "project.yaml"
        lineage = [
            "global_1", "preprocess_1", "cc_1", "filter_1", "stack_1"
        ]
        params = build_params_from_project_yaml(yaml_path, lineage)
        assert params.lineage_names == lineage
        assert params.global_.startdate == "2014-01-01"
        assert params.preprocess.cc_sampling_rate == 50.0
        assert params.stack.stack_method == "linear"

    def test_meta_keys_stripped(self, simple_project):
        """'after' and 'next_steps' must not appear as config values."""
        yaml_path = simple_project / "project.yaml"
        lineage = ["global_1", "preprocess_1", "cc_1", "filter_1", "stack_1"]
        params = build_params_from_project_yaml(yaml_path, lineage)
        layers = object.__getattribute__(params, "_layers")
        for cat, layer in layers.items():
            assert "after" not in layer, f"'after' leaked into {cat} layer"
            assert "next_steps" not in layer, f"'next_steps' leaked into {cat} layer"

    def test_missing_step_gives_empty_layer(self, simple_project):
        """Steps absent from project.yaml get an empty layer (graceful)."""
        yaml_path = simple_project / "project.yaml"
        lineage = ["global_1", "nonexistent_step_1"]
        params = build_params_from_project_yaml(yaml_path, lineage)
        assert params.lineage_names == lineage

    def test_filter_layer(self, simple_project):
        yaml_path = simple_project / "project.yaml"
        lineage = ["global_1", "preprocess_1", "cc_1", "filter_2", "stack_2"]
        params = build_params_from_project_yaml(yaml_path, lineage)
        assert params.filter.freqmin == 1.0
        assert params.filter.freqmax == 5.0


# ---------------------------------------------------------------------------
# Unit tests — LEVEL_GLOBS / LEVEL_CATEGORIES consistency
# ---------------------------------------------------------------------------

class TestLevelConstants:
    def test_same_keys(self):
        assert set(LEVEL_GLOBS) == set(LEVEL_CATEGORIES)

    def test_all_levels_have_categories(self):
        for level, cats in LEVEL_CATEGORIES.items():
            assert cats, f"LEVEL_CATEGORIES[{level!r}] is empty"

    def test_wavelet_globs_match_category_names(self):
        # Wavelet dirs are named wavelet_N / wavelet_dtt_N, not wct_*
        for pattern in LEVEL_GLOBS["wavelet"]:
            assert "wct_" not in pattern, (
                f"LEVEL_GLOBS['wavelet'] still references wct_*: {pattern!r}"
            )

    def test_dvv_pattern_covers_all_dvv_cats(self):
        dvv_cats = LEVEL_CATEGORIES["dvv"]
        assert all("dvv" in c for c in dvv_cats)


# ---------------------------------------------------------------------------
# Integration tests — export → extract → MSNoiseProject.list()
# ---------------------------------------------------------------------------

class TestExportImportRoundtrip:
    def test_export_creates_archive(self, simple_project):
        out = simple_project / "level_stack.tar.zst"
        sha = export_project(simple_project, "stack", out)
        assert out.exists()
        assert len(sha) == 64  # hex sha256

    def test_archive_contains_expected_members(self, simple_project):
        import zstandard as zstd

        out = simple_project / "level_stack.tar.zst"
        export_project(simple_project, "stack", out)

        with open(out, "rb") as fh:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(fh) as r:
                with tarfile.open(fileobj=r, mode="r|") as tf:
                    names = [m.name for m in tf]

        assert "meta.yaml" in names
        assert "MANIFEST.json" in names
        assert "project.yaml" in names
        # params.yaml written alongside each step dir
        assert any("stack_1/params.yaml" in n for n in names)
        assert any("refstack_1/params.yaml" in n for n in names)
        # actual output files
        assert any("_output" in n and n.endswith(".nc") for n in names)

    def test_manifest_sha256_correct(self, simple_project):
        import zstandard as zstd

        out = simple_project / "level_stack.tar.zst"
        export_project(simple_project, "stack", out)

        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            with open(out, "rb") as fh:
                dctx = zstd.ZstdDecompressor()
                with dctx.stream_reader(fh) as r:
                    with tarfile.open(fileobj=r, mode="r|") as tf:
                        tf.extractall(td)

            manifest = json.loads((td / "MANIFEST.json").read_text())
            for rel_path, info in manifest.items():
                actual = file_sha256(td / rel_path)
                assert actual == info["sha256"], f"sha256 mismatch for {rel_path}"

    def test_meta_yaml_content(self, simple_project):
        import zstandard as zstd

        out = simple_project / "level_stack.tar.zst"
        export_project(simple_project, "stack", out)

        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            with open(out, "rb") as fh:
                dctx = zstd.ZstdDecompressor()
                with dctx.stream_reader(fh) as r:
                    with tarfile.open(fileobj=r, mode="r|") as tf:
                        tf.extractall(td)

            meta = yaml.safe_load((td / "meta.yaml").read_text())
            assert meta["entry_level"] == "stack"
            assert "msnoise_version" in meta
            assert "created_at" in meta

    def test_extract_and_list(self, simple_project):
        out = simple_project / "level_stack.tar.zst"
        export_project(simple_project, "stack", out)

        with tempfile.TemporaryDirectory() as td:
            root = extract_archive(out, td)
            assert (root / "project.yaml").exists()

            proj = MSNoiseProject.from_project_dir(root)
            results = proj.list("stack")

            assert len(results) == 2
            categories = {r.category for r in results}
            assert categories == {"stack"}
            lineages = [r.lineage_names[-1] for r in results]
            assert "stack_1" in lineages
            assert "stack_2" in lineages

    def test_params_yaml_written_on_first_list(self, simple_project):
        out = simple_project / "level_stack.tar.zst"
        export_project(simple_project, "stack", out)

        with tempfile.TemporaryDirectory() as td:
            root = extract_archive(out, td)
            proj = MSNoiseProject.from_project_dir(root)
            proj.list("stack")  # triggers params.yaml creation

            for cat in ("stack_1", "stack_2"):
                params_path = (
                    root
                    / "global_1" / "preprocess_1" / "cc_1"
                    / f"filter_{cat[-1]}" / cat / "params.yaml"
                )
                assert params_path.exists(), f"params.yaml missing for {cat}"

    def test_params_yaml_reused_on_second_list(self, simple_project):
        out = simple_project / "level_stack.tar.zst"
        export_project(simple_project, "stack", out)

        with tempfile.TemporaryDirectory() as td:
            root = extract_archive(out, td)
            proj = MSNoiseProject.from_project_dir(root)
            proj.list("stack")

            # Record mtime of params.yaml after first call
            params_path = (
                root / "global_1" / "preprocess_1" / "cc_1"
                / "filter_1" / "stack_1" / "params.yaml"
            )
            mtime_1 = params_path.stat().st_mtime

            proj.list("stack")  # second call — should reuse
            mtime_2 = params_path.stat().st_mtime
            assert mtime_1 == mtime_2, "params.yaml was re-written on second list()"

    def test_output_folder_is_project_root(self, simple_project):
        """MSNoiseResult.output_folder must point at project root, not step dir."""
        out = simple_project / "level_stack.tar.zst"
        export_project(simple_project, "stack", out)

        with tempfile.TemporaryDirectory() as td:
            root = extract_archive(out, td)
            proj = MSNoiseProject.from_project_dir(root)
            results = proj.list("stack")

            for r in results:
                assert r.output_folder == str(root), (
                    f"output_folder {r.output_folder!r} ≠ project root {root!r}"
                )

    def test_refstack_results_discoverable(self, simple_project):
        out = simple_project / "level_stack.tar.zst"
        export_project(simple_project, "stack", out)

        with tempfile.TemporaryDirectory() as td:
            root = extract_archive(out, td)
            proj = MSNoiseProject.from_project_dir(root)
            results = proj.list("refstack")
            assert len(results) == 2

    def test_from_project_dir_raises_without_project_yaml(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="project.yaml"):
            MSNoiseProject.from_project_dir(tmp_path)

    def test_db_property_raises_without_init(self, simple_project):
        proj = MSNoiseProject.from_project_dir(simple_project)
        with pytest.raises(RuntimeError, match="init_db"):
            _ = proj.db
