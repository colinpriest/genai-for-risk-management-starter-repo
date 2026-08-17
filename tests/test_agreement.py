"""Tests for the ICC(2,1) helper.

The point estimate is checked against cases with a known answer, so a wrong implementation
cannot pass. Perfect agreement must give 1.0; pure noise must give roughly 0; a constant
per-rater offset must reduce ICC(2,1) (absolute agreement) while leaving correlation at 1.0 —
that last one is the property that makes ICC the right statistic here rather than Pearson r.
"""
from __future__ import annotations
import numpy as np
import pytest

from src.agreement import ai_vs_humans, human_range, icc21, icc21_ci


class TestICC21KnownCases:
    def test_perfect_agreement_is_one(self):
        col = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        assert icc21(np.column_stack([col, col, col])) == pytest.approx(1.0, abs=1e-9)

    def test_pure_noise_is_near_zero(self):
        rng = np.random.default_rng(0)
        vals = [icc21(rng.random((30, 3))) for _ in range(40)]
        assert abs(float(np.mean(vals))) < 0.15

    def test_constant_offset_penalised_though_correlation_is_perfect(self):
        """The reason to use ICC rather than Pearson r.

        Rater B agrees perfectly on ORDER but is 0.2 low on every document. Correlation says
        1.0; absolute agreement should not.
        """
        a = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        m = np.column_stack([a, a - 0.2])
        assert np.corrcoef(m[:, 0], m[:, 1])[0, 1] == pytest.approx(1.0)
        assert icc21(m) < 0.95

    def test_larger_offset_penalised_more(self):
        a = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        assert icc21(np.column_stack([a, a - 0.4])) < icc21(np.column_stack([a, a - 0.1]))

    def test_no_between_document_variance_gives_no_reliability(self):
        """Every document scored the same on average: nothing to be reliable about."""
        m = np.array([[0.5, 0.6], [0.6, 0.5], [0.5, 0.6], [0.6, 0.5]])
        assert icc21(m) < 0.1

    def test_invariant_to_document_order(self):
        rng = np.random.default_rng(3)
        m = rng.random((20, 3))
        assert icc21(m) == pytest.approx(icc21(m[rng.permutation(20)]))

    def test_invariant_to_rater_order(self):
        rng = np.random.default_rng(4)
        m = rng.random((20, 3))
        assert icc21(m) == pytest.approx(icc21(m[:, [2, 0, 1]]))


class TestICC21Validation:
    def test_rejects_missing_values(self):
        m = np.array([[0.5, 0.6], [np.nan, 0.5], [0.4, 0.4]])
        with pytest.raises(ValueError, match="complete matrix"):
            icc21(m)

    def test_rejects_single_rater(self):
        with pytest.raises(ValueError, match="at least 2"):
            icc21(np.array([[0.5], [0.6], [0.7]]))

    def test_rejects_single_document(self):
        with pytest.raises(ValueError, match="at least 2"):
            icc21(np.array([[0.5, 0.6]]))

    def test_rejects_one_dimensional(self):
        with pytest.raises(ValueError, match="2-D"):
            icc21(np.array([0.5, 0.6, 0.7]))


class TestBootstrapInterval:
    def test_interval_brackets_the_point_estimate(self):
        rng = np.random.default_rng(1)
        truth = rng.random(20)
        m = np.column_stack([truth + rng.normal(0, 0.05, 20) for _ in range(3)])
        r = icc21_ci(m, n_boot=500)
        assert r["ci_low"] <= r["icc"] <= r["ci_high"]

    def test_interval_is_reproducible_under_a_fixed_seed(self):
        rng = np.random.default_rng(2)
        m = rng.random((20, 3))
        assert icc21_ci(m, n_boot=300, seed=7) == icc21_ci(m, n_boot=300, seed=7)

    def test_more_documents_narrows_the_interval(self):
        rng = np.random.default_rng(5)
        def width(n):
            truth = rng.random(n)
            m = np.column_stack([truth + rng.normal(0, 0.1, n) for _ in range(3)])
            r = icc21_ci(m, n_boot=400, seed=11)
            return r["ci_high"] - r["ci_low"]
        assert width(120) < width(15)

    def test_twenty_document_interval_is_honestly_wide(self):
        """The protocol's own sample size. If this ever gets narrow, suspect the resampler."""
        rng = np.random.default_rng(6)
        truth = rng.random(20)
        m = np.column_stack([truth + rng.normal(0, 0.15, 20) for _ in range(3)])
        r = icc21_ci(m, n_boot=800, seed=3)
        assert r["ci_high"] - r["ci_low"] > 0.10

    def test_reports_interpretation_band(self):
        col = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 0.2, 0.8])
        assert icc21_ci(np.column_stack([col, col]), n_boot=200)["interpretation"] == "excellent"


class TestHumanRange:
    def test_range_spans_rater_means(self):
        m = np.column_stack([np.full(5, 0.2), np.full(5, 0.5), np.full(5, 0.8)])
        r = human_range(m)
        assert r["low"] == pytest.approx(0.2)
        assert r["high"] == pytest.approx(0.8)
        assert r["spread"] == pytest.approx(0.6)

    def test_ai_inside_range_detected(self):
        m = np.column_stack([np.full(6, 0.3), np.full(6, 0.7)])
        assert ai_vs_humans(m, np.full(6, 0.5))["ai_inside_human_range"]

    def test_ai_outside_range_detected(self):
        m = np.column_stack([np.full(6, 0.3), np.full(6, 0.7)])
        assert not ai_vs_humans(m, np.full(6, 0.95))["ai_inside_human_range"]

    def test_ai_comparison_rejects_length_mismatch(self):
        m = np.column_stack([np.full(6, 0.3), np.full(6, 0.7)])
        with pytest.raises(ValueError, match="rows"):
            ai_vs_humans(m, np.full(4, 0.5))

    def test_ai_comparison_reports_human_human_first(self):
        rng = np.random.default_rng(8)
        m = rng.random((15, 3))
        out = ai_vs_humans(m, rng.random(15))
        assert "human_human" in out and "icc" in out["human_human"]
        assert list(out)[0] == "human_human"
