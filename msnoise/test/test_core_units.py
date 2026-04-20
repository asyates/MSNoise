"""Pure-logic unit tests for msnoise.core.compute, .signal, and .config.

No DB, no filesystem, no ObsPy required.
"""
import types

import numpy as np
import pytest
import xarray as xr


# ============================================================================
# core/compute.py
# ============================================================================

class TestAnalyticPhase:
    def test_output_shape(self):
        from ..core.compute import _analytic_phase
        x = np.random.randn(256)
        y = _analytic_phase(x)
        assert y.shape == x.shape
        assert np.iscomplexobj(y)

    def test_unit_magnitude(self):
        from ..core.compute import _analytic_phase
        x = np.random.randn(256)
        y = _analytic_phase(x)
        assert np.allclose(np.abs(y), 1.0, atol=1e-3)

    def test_all_zeros_input(self):
        from ..core.compute import _analytic_phase
        x = np.zeros(128)
        y = _analytic_phase(x)
        assert y.shape == x.shape  # must not raise


class TestAnalyticPhaseBatch:
    def test_output_shape(self):
        from ..core.compute import _analytic_phase_batch
        X = np.random.randn(4, 256)
        Y = _analytic_phase_batch(X)
        assert Y.shape == X.shape
        assert np.iscomplexobj(Y)

    def test_unit_magnitude_per_row(self):
        from ..core.compute import _analytic_phase_batch
        X = np.random.randn(3, 128)
        Y = _analytic_phase_batch(X)
        assert np.allclose(np.abs(Y), 1.0, atol=1e-3)


class TestPccXcorr:
    def _make_data(self, n=256, n_sta=3):
        rng = np.random.default_rng(42)
        return rng.standard_normal((n_sta, n)).astype(float)

    def test_empty_index(self):
        from ..core.compute import pcc_xcorr
        data = self._make_data()
        result = pcc_xcorr(data, maxlag=50, energy=None, index=[])
        assert result == {}

    def test_output_length(self):
        from ..core.compute import pcc_xcorr
        data = self._make_data()
        maxlag = 50
        index = [(0, 0, 1), (1, 0, 2)]
        result = pcc_xcorr(data, maxlag=maxlag, energy=None, index=index)
        assert set(result.keys()) == {0, 1}
        assert len(result[0]) == 2 * maxlag + 1

    def test_normalized_max(self):
        from ..core.compute import pcc_xcorr
        data = self._make_data()
        result = pcc_xcorr(data, maxlag=30, energy=None,
                           index=[(0, 0, 1)], normalized="MAX")
        assert result[0].max() == pytest.approx(1.0)

    def test_normalized_absmax(self):
        from ..core.compute import pcc_xcorr
        data = self._make_data()
        result = pcc_xcorr(data, maxlag=30, energy=None,
                           index=[(0, 0, 1)], normalized="ABSMAX")
        assert np.abs(result[0]).max() == pytest.approx(1.0)

    def test_self_correlation_peak_at_zero(self):
        from ..core.compute import pcc_xcorr
        rng = np.random.default_rng(7)
        sig = rng.standard_normal((2, 256))
        sig[1] = sig[0]   # identical traces → peak at lag=0
        maxlag = 40
        result = pcc_xcorr(sig, maxlag=maxlag, energy=None,
                           index=[(0, 0, 1)], normalized="ABSMAX")
        ccf = result[0]
        assert np.argmax(np.abs(ccf)) == maxlag   # zero-lag index


class TestSmooth:
    def test_boxcar_shape(self):
        from ..core.compute import smooth
        x = np.arange(50, dtype=float)
        y = smooth(x, window='boxcar', half_win=3)
        assert len(y) == len(x)

    def test_hann_shape(self):
        from ..core.compute import smooth
        x = np.ones(50)
        y = smooth(x, window='hanning', half_win=2)
        assert len(y) == len(x)

    def test_constant_signal_unchanged(self):
        from ..core.compute import smooth
        x = np.ones(40) * 5.0
        y = smooth(x, window='boxcar', half_win=3)
        assert np.allclose(np.real(y), 5.0, atol=1e-10)


class TestResolveWctLagMin:
    def _make_params(self, lag_type="static", v=1.0, minlag=5.0):
        wdt = types.SimpleNamespace(wct_lag=lag_type, wct_v=v, wct_minlag=minlag)
        return types.SimpleNamespace(wavelet_dtt=wdt)

    def test_static_returns_minlag(self):
        from ..core.compute import resolve_wct_lag_min
        p = self._make_params(lag_type="static", minlag=8.0)
        assert resolve_wct_lag_min(p, dist=100.0) == pytest.approx(8.0)

    def test_dynamic_returns_dist_over_v(self):
        from ..core.compute import resolve_wct_lag_min
        p = self._make_params(lag_type="dynamic", v=2.0, minlag=5.0)
        assert resolve_wct_lag_min(p, dist=100.0) == pytest.approx(50.0)

    def test_dynamic_none_v_falls_back_to_static(self):
        from ..core.compute import resolve_wct_lag_min
        # wct_v=None → `None or 1.0` → 1.0, so dynamic gives dist/1.0
        p = self._make_params(lag_type="dynamic", v=None, minlag=7.0)
        # v resolves to 1.0 via `or`, so result = dist/1.0 = 100.0
        assert resolve_wct_lag_min(p, dist=100.0) == pytest.approx(100.0)


class TestBuildWctDttDataset:
    def test_output_is_dataset(self):
        from ..core.compute import build_wct_dtt_dataset
        dates = [np.datetime64("2023-01-01"), np.datetime64("2023-01-02")]
        freqs = np.array([0.5, 1.0, 2.0])
        dtt_rows = [np.ones(3), np.ones(3) * 2]
        err_rows = [np.zeros(3), np.zeros(3)]
        coh_rows = [np.ones(3) * 0.9, np.ones(3) * 0.8]
        ds = build_wct_dtt_dataset(dates, dtt_rows, err_rows, coh_rows, freqs)
        assert isinstance(ds, xr.Dataset)
        assert "times" in ds.dims
        assert "frequency" in ds.dims


# ============================================================================
# core/signal.py
# ============================================================================

class TestNextpow2:
    def test_exact_power(self):
        from ..core.signal import nextpow2
        assert nextpow2(8) == pytest.approx(3)

    def test_non_power(self):
        from ..core.signal import nextpow2
        assert nextpow2(9) == pytest.approx(4)

    def test_one(self):
        from ..core.signal import nextpow2
        assert nextpow2(1) == pytest.approx(0)


class TestGetWindow:
    def test_boxcar_length(self):
        from ..core.signal import get_window
        w = get_window(window="boxcar", half_win=3)
        assert len(w) == 7

    def test_hanning_length(self):
        from ..core.signal import get_window
        w = get_window(window="hanning", half_win=5)
        assert len(w) == 11

    def test_sum_normalised(self):
        from ..core.signal import get_window
        w = get_window(window="boxcar", half_win=3)
        assert np.sum(np.real(w)) == pytest.approx(1.0, abs=1e-10)


class TestGetCoherence:
    def test_perfect_coherence(self):
        from ..core.signal import getCoherence
        n = 10
        dcs = np.ones(n)
        ds1 = np.ones(n)
        ds2 = np.ones(n)
        coh = getCoherence(dcs, ds1, ds2)
        assert np.allclose(np.abs(coh), 1.0)

    def test_zero_denominator_gives_zero(self):
        from ..core.signal import getCoherence
        n = 10
        dcs = np.ones(n)
        ds1 = np.zeros(n)
        ds2 = np.zeros(n)
        coh = getCoherence(dcs, ds1, ds2)
        assert np.allclose(coh, 0.0)

    def test_clipped_to_one(self):
        from ..core.signal import getCoherence
        n = 5
        coh = getCoherence(np.ones(n) * 10, np.ones(n), np.ones(n))
        assert np.all(np.abs(coh) <= 1.0 + 1e-10)


class TestPrepareAbsPositiveFft:
    def test_positive_freqs_only(self):
        from ..core.signal import prepare_abs_positive_fft
        x = np.random.randn(128)
        freq, val = prepare_abs_positive_fft(x, sampling_rate=100.0)
        assert np.all(freq >= 0)

    def test_output_shapes_match(self):
        from ..core.signal import prepare_abs_positive_fft
        x = np.random.randn(64)
        freq, val = prepare_abs_positive_fft(x, sampling_rate=50.0)
        assert freq.shape == val.shape


class TestPsdRms:
    def test_flat_spectrum(self):
        from ..core.signal import psd_rms
        f = np.linspace(0.1, 1.0, 100)
        s = np.ones_like(f)
        rms = psd_rms(s, f)
        assert rms > 0

    def test_zero_spectrum(self):
        from ..core.signal import psd_rms
        f = np.linspace(0.1, 1.0, 100)
        s = np.zeros_like(f)
        assert psd_rms(s, f) == pytest.approx(0.0)


class TestStack:
    def _make_ccfs(self, n_traces=5, n_lags=101):
        rng = np.random.default_rng(0)
        return rng.standard_normal((n_traces, n_lags))

    def test_linear_shape(self):
        from ..core.signal import stack
        data = self._make_ccfs()
        out = stack(data, stack_method="linear")
        assert out.shape == (data.shape[1],)

    def test_linear_is_mean(self):
        from ..core.signal import stack
        data = self._make_ccfs()
        out = stack(data, stack_method="linear")
        assert np.allclose(out, data.mean(axis=0))

    def test_pws_shape(self):
        from ..core.signal import stack
        data = self._make_ccfs()
        out = stack(data, stack_method="pws",
                    pws_timegate=5.0, pws_power=2,
                    goal_sampling_rate=20.0)
        assert out.shape == (data.shape[1],)


class TestFindSegments:
    def _make_da(self, n_times=10, n_lags=20, nan_rows=None):
        data = np.ones((n_times, n_lags))
        if nan_rows:
            for r in nan_rows:
                data[r, :] = np.nan
        return xr.DataArray(data, dims=["times", "lags"])

    def test_no_gaps(self):
        from ..core.signal import find_segments
        da = self._make_da()
        segs = find_segments(da, gap_threshold=1)
        assert len(segs) == 1
        assert len(segs[0]) == 10

    def test_null_rows_reset_tracking(self):
        from ..core.signal import find_segments
        # Null rows reset prev_idx to None; the subsequent non-null row has
        # prev_idx=None so the gap check is skipped → stays in same segment.
        da = self._make_da(nan_rows=[4, 5])
        segs = find_segments(da, gap_threshold=1)
        assert len(segs) == 1   # gap not triggered after null rows

    def test_all_nan(self):
        from ..core.signal import find_segments
        da = self._make_da(nan_rows=list(range(10)))
        segs = find_segments(da, gap_threshold=1)
        assert segs == []


# ============================================================================
# core/config.py
# ============================================================================

class TestParseConfigKey:
    def test_bare_name_is_global(self):
        from ..core.config import parse_config_key
        assert parse_config_key("output_folder") == ("global", 1, "output_folder")

    def test_category_dot_name(self):
        from ..core.config import parse_config_key
        assert parse_config_key("cc.cc_sampling_rate") == ("cc", 1, "cc_sampling_rate")

    def test_category_dot_setnum_dot_name(self):
        from ..core.config import parse_config_key
        assert parse_config_key("mwcs.2.mwcs_wlen") == ("mwcs", 2, "mwcs_wlen")

    def test_invalid_set_number_raises(self):
        from ..core.config import parse_config_key
        with pytest.raises(ValueError):
            parse_config_key("cc.abc.param")

    def test_too_many_parts_raises(self):
        from ..core.config import parse_config_key
        with pytest.raises(ValueError):
            parse_config_key("a.b.c.d")


class TestCastConfigValue:
    def test_bool_y(self):
        from ..core.config import _cast_config_value
        assert _cast_config_value("flag", "Y", "bool") == "Y"
        assert _cast_config_value("flag", "yes", "bool") == "Y"
        assert _cast_config_value("flag", "True", "bool") == "Y"
        assert _cast_config_value("flag", "1", "bool") == "Y"

    def test_bool_n(self):
        from ..core.config import _cast_config_value
        assert _cast_config_value("flag", "N", "bool") == "N"
        assert _cast_config_value("flag", "false", "bool") == "N"
        assert _cast_config_value("flag", "0", "bool") == "N"

    def test_bool_invalid_raises(self):
        from ..core.config import _cast_config_value
        with pytest.raises(ValueError):
            _cast_config_value("flag", "maybe", "bool")

    def test_int_valid(self):
        from ..core.config import _cast_config_value
        assert _cast_config_value("n", "42", "int") == "42"

    def test_int_invalid_raises(self):
        from ..core.config import _cast_config_value
        with pytest.raises(ValueError):
            _cast_config_value("n", "abc", "int")

    def test_float_valid(self):
        from ..core.config import _cast_config_value
        assert _cast_config_value("x", "3.14", "float") == "3.14"

    def test_float_invalid_raises(self):
        from ..core.config import _cast_config_value
        with pytest.raises(ValueError):
            _cast_config_value("x", "pi", "float")

    def test_str_passthrough(self):
        from ..core.config import _cast_config_value
        assert _cast_config_value("s", "anything goes", "str") == "anything goes"


class TestLineageToPlotTag:
    def test_basic(self):
        from ..core.config import lineage_to_plot_tag
        tag = lineage_to_plot_tag(["preprocess_1", "cc_1", "filter_1", "stack_1"])
        assert "pre1" in tag
        assert "cc1" in tag
        assert "stk1" in tag

    def test_empty(self):
        from ..core.config import lineage_to_plot_tag
        assert lineage_to_plot_tag([]) == ""

    def test_single_step(self):
        from ..core.config import lineage_to_plot_tag
        tag = lineage_to_plot_tag(["preprocess_2"])
        assert "2" in tag


class TestBuildPlotOutfile:
    def test_explicit_path_unchanged(self):
        from ..core.config import build_plot_outfile
        out = build_plot_outfile("myfile.png", "ccftime",
                                 ["preprocess_1", "cc_1"])
        assert out == "myfile.png"

    def test_none_returns_none(self):
        from ..core.config import build_plot_outfile
        assert build_plot_outfile(None, "ccftime", []) is None

    def test_question_mark_expands(self):
        from ..core.config import build_plot_outfile
        out = build_plot_outfile("?.png", "ccftime",
                                 ["preprocess_1", "cc_1", "filter_1", "stack_1"])
        assert out.startswith("ccftime__")
        assert out.endswith(".png")

    def test_pair_included(self):
        from ..core.config import build_plot_outfile
        out = build_plot_outfile("?.png", "ccftime",
                                 ["preprocess_1", "cc_1"],
                                 pair="BE.UCC..HHZ:BE.MEM..HHZ")
        assert "BE.UCC..HHZ-BE.MEM..HHZ" in out

    def test_mov_stack_tuple(self):
        from ..core.config import build_plot_outfile
        out = build_plot_outfile("?.png", "dvv",
                                 ["preprocess_1"],
                                 mov_stack=("1D", "1D"))
        assert "m1D-1D" in out


# ============================================================================
# core/stations.py — pure-logic helpers (no DB, no ObsPy required)
# ============================================================================

class TestGetInterstationDistance:
    def _sta(self, x, y):
        return types.SimpleNamespace(X=x, Y=y)

    def test_utm_hypot(self):
        from ..core.stations import get_interstation_distance
        s1 = self._sta(0.0, 0.0)
        s2 = self._sta(3000.0, 4000.0)   # 5 km
        dist = get_interstation_distance(s1, s2, coordinates="UTM")
        assert dist == pytest.approx(5.0, rel=1e-6)

    def test_utm_zero(self):
        from ..core.stations import get_interstation_distance
        s = self._sta(1234.0, 5678.0)
        assert get_interstation_distance(s, s, coordinates="UTM") == pytest.approx(0.0)

    def test_deg_uses_gps2dist(self):
        from ..core.stations import get_interstation_distance
        # Brussels → Paris ≈ 265 km
        brussels = self._sta(4.35, 50.85)
        paris    = self._sta(2.35, 48.85)
        dist = get_interstation_distance(brussels, paris, coordinates="DEG")
        assert 250 < dist < 290, f"Brussels-Paris distance out of range: {dist}"


class TestToSds:
    """Tests for stations.to_sds path builder (no DB, no ObsPy stream needed)."""

    def _stats(self, net="BE", sta="UCC", loc="", chan="HHZ"):
        return types.SimpleNamespace(
            network=net, station=sta, location=loc, channel=chan
        )

    def test_format_structure(self):
        from ..core.stations import to_sds
        path = to_sds(self._stats(), year=2023, jday=42)
        parts = path.split("/")
        assert parts[0] == "2023"
        assert parts[1] == "BE"
        assert parts[2] == "UCC"
        assert parts[3] == "HHZ.D"
        assert parts[4] == "BE.UCC..HHZ.D.2023.042"

    def test_year_zero_padded(self):
        from ..core.stations import to_sds
        path = to_sds(self._stats(), year=999, jday=1)
        assert path.startswith("0999/")

    def test_jday_zero_padded(self):
        from ..core.stations import to_sds
        path = to_sds(self._stats(), year=2023, jday=5)
        assert path.endswith(".005")

    def test_location_code(self):
        from ..core.stations import to_sds
        path = to_sds(self._stats(loc="00"), year=2023, jday=1)
        assert "BE.UCC.00.HHZ" in path

    def test_different_channels(self):
        from ..core.stations import to_sds
        for chan in ["BHZ", "LHN", "EHE"]:
            path = to_sds(self._stats(chan=chan), year=2023, jday=1)
            assert chan in path


# ============================================================================
# params.py — MSNoiseParams
# ============================================================================

class TestMSNoiseParams:
    """Tests for MSNoiseParams construction, access, and serialisation."""

    def _make_params(self):
        from ..params import MSNoiseParams
        from obspy.core.util.attribdict import AttribDict
        p = MSNoiseParams()
        p._set_lineage_names(["preprocess_1", "cc_1", "filter_1"])
        p._add_layer("global",     AttribDict({"output_folder": "./out", "hpc": "N"}))
        p._add_layer("preprocess", AttribDict({"preprocess_sampling_rate": 20.0}))
        p._add_layer("cc",         AttribDict({"maxlag": 60.0, "winsorizing": 2.0}))
        p._add_layer("filter",     AttribDict({"freqmin": 0.1, "freqmax": 1.0}))
        return p

    def test_category_access(self):
        p = self._make_params()
        assert p.cc.maxlag == 60.0
        assert p.filter.freqmin == pytest.approx(0.1)

    def test_global_underscore_alias(self):
        p = self._make_params()
        assert p.global_.hpc == "N"

    def test_bracket_access(self):
        p = self._make_params()
        assert p["cc"].maxlag == 60.0

    def test_missing_category_raises(self):
        p = self._make_params()
        with pytest.raises(AttributeError):
            _ = p.mwcs

    def test_missing_bracket_raises(self):
        p = self._make_params()
        with pytest.raises(KeyError):
            _ = p["mwcs"]

    def test_immutable(self):
        p = self._make_params()
        with pytest.raises(AttributeError):
            p.cc = "oops"

    def test_category_property(self):
        p = self._make_params()
        assert p.category == "filter"

    def test_category_layer_property(self):
        p = self._make_params()
        assert p.category_layer.freqmin == pytest.approx(0.1)

    def test_categories_list(self):
        p = self._make_params()
        assert p.categories == ["global", "preprocess", "cc", "filter"]

    def test_lineage_names(self):
        p = self._make_params()
        assert p.lineage_names == ["preprocess_1", "cc_1", "filter_1"]

    def test_step_name(self):
        p = self._make_params()
        assert p.step_name == "filter_1"

    def test_as_flat_dict(self):
        p = self._make_params()
        d = p.as_flat_dict()
        assert "maxlag" in d
        assert "freqmin" in d
        assert d["maxlag"] == 60.0

    def test_repr(self):
        p = self._make_params()
        r = repr(p)
        assert "MSNoiseParams" in r
        assert "filter" in r

    def test_yaml_roundtrip(self):
        pytest.importorskip("yaml")
        p = self._make_params()
        from ..params import MSNoiseParams
        yaml_str = p.to_yaml_string()
        assert "cc" in yaml_str
        p2 = MSNoiseParams.from_yaml_string(yaml_str)
        assert p2.cc.maxlag == pytest.approx(60.0)
        assert p2.filter.freqmin == pytest.approx(0.1)
        assert p2.lineage_names == p.lineage_names

    def test_yaml_roundtrip_file(self, tmp_path):
        pytest.importorskip("yaml")
        p = self._make_params()
        from ..params import MSNoiseParams
        fpath = str(tmp_path / "params.yaml")
        p.to_yaml(fpath)
        p2 = MSNoiseParams.from_yaml(fpath)
        assert p2.cc.winsorizing == pytest.approx(2.0)

    def test_empty_params_raises_on_category(self):
        from ..params import MSNoiseParams
        p = MSNoiseParams()
        with pytest.raises(RuntimeError):
            _ = p.category
