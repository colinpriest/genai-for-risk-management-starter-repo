"""Unit tests for the regime-model mechanics that fail SILENTLY when they are wrong.

None of these need a fitted model, saved data, or an API key — they run on matrices written
out by hand, so they run from Week 1 and they run in a second.

Each test here corresponds to a bug that shipped in an earlier version of this repository.
All three produced plausible-looking output and wrong conclusions:

  1. A 63-trading-day horizon advanced the chain 3 times instead of 63, because the code
     assumed a monthly transition matrix. There is no monthly matrix. Reported quarter-ahead
     stress of 0.2% where the correct figure was 15%.
  2. The saved transition matrix was left in raw statsmodels state order while the regime
     names beside it were volatility-ordered, so every probability was attached to the wrong
     regime. Long-run "stressed" came out at 44% against an observed 21%.
  3. Coefficients were looked up by feature name. statsmodels names them positionally, so
     the lookup returned zero and every scenario produced an identical answer.
"""
from __future__ import annotations
import numpy as np
import pytest

from src.stage4_regime_model import ordered_index, propagate, stationary


# A two-state chain with arithmetic you can check by hand. Column-stochastic: columns are
# "from", rows are "to". Stay probabilities 0.9 and 0.8, so the stationary distribution is
# (2/3, 1/3) and expected durations are 10 and 5 days.
P2 = np.array([[0.9, 0.2],
               [0.1, 0.8]])


class TestPropagate:
    def test_zero_steps_is_the_identity(self):
        p0 = np.array([1.0, 0.0])
        assert np.allclose(propagate(P2, p0, 0), p0)

    def test_one_step_matches_hand_calculation(self):
        # Starting certain in state 0: after one step, 0.9 stays, 0.1 moves.
        assert np.allclose(propagate(P2, np.array([1.0, 0.0]), 1), [0.9, 0.1])

    def test_n_steps_equals_n_applications(self):
        """The bug: horizon_days // 21 advanced a DAILY matrix 3 times for a 63-day horizon."""
        p0 = np.array([1.0, 0.0])
        manual = p0.copy()
        for _ in range(63):
            manual = P2 @ manual
        assert np.allclose(propagate(P2, p0, 63), manual)

    def test_63_steps_differs_materially_from_3(self):
        """Guard the specific regression, not just the general rule."""
        p0 = np.array([1.0, 0.0])
        assert abs(propagate(P2, p0, 63)[1] - propagate(P2, p0, 3)[1]) > 0.05

    def test_long_horizon_converges_to_stationary(self):
        for start in ([1.0, 0.0], [0.0, 1.0], [0.5, 0.5]):
            assert np.allclose(propagate(P2, np.array(start), 5000), [2 / 3, 1 / 3], atol=1e-6)

    def test_result_is_a_distribution(self):
        p = propagate(P2, np.array([0.3, 0.7]), 17)
        assert np.isclose(p.sum(), 1.0) and (p >= 0).all()

    def test_rejects_row_stochastic_input(self):
        """Passing the transposed matrix silently gives wrong answers, so it must raise."""
        with pytest.raises(ValueError, match="COLUMN-stochastic"):
            propagate(P2.T, np.array([1.0, 0.0]), 10)

    def test_rejects_negative_horizon(self):
        with pytest.raises(ValueError):
            propagate(P2, np.array([1.0, 0.0]), -1)


class TestStationary:
    def test_two_state_stationary_is_exact(self):
        assert np.allclose(stationary(P2), [2 / 3, 1 / 3])

    def test_stationary_is_a_fixed_point(self):
        s = stationary(P2)
        assert np.allclose(P2 @ s, s)

    def test_three_state_stationary_is_a_fixed_point(self):
        P3 = np.array([[0.97, 0.03, 0.00],
                       [0.03, 0.94, 0.05],
                       [0.00, 0.03, 0.95]])
        s = stationary(P3)
        assert np.isclose(s.sum(), 1.0)
        assert np.allclose(P3 @ s, s, atol=1e-9)


class TestOrderedIndex:
    def test_inverts_the_remap(self):
        # remap is {raw_state: ordered_slot}. This one reverses a 3-state model, which is
        # exactly what the fitted Australian model did.
        assert ordered_index({0: 2, 1: 1, 2: 0}) == [2, 1, 0]

    def test_identity_remap(self):
        assert ordered_index({0: 0, 1: 1, 2: 2}) == [0, 1, 2]

    def test_round_trips_against_the_matrix_reordering(self):
        """Reordering by ordered_index must move persistence onto the right diagonal entry."""
        raw = np.array([[0.80, 0.05, 0.00],
                        [0.20, 0.90, 0.30],
                        [0.00, 0.05, 0.70]])
        remap = {0: 2, 1: 1, 2: 0}          # raw state 2 is the calmest
        inv = ordered_index(remap)
        ordered = raw[np.ix_(inv, inv)]
        assert np.allclose(np.diag(ordered), [0.70, 0.90, 0.80])
        assert np.allclose(ordered.sum(axis=0), 1.0)

    def test_reordering_preserves_stationary_mass(self):
        """A permutation must permute the stationary distribution, not change it."""
        raw = np.array([[0.80, 0.05, 0.00],
                        [0.20, 0.90, 0.30],
                        [0.00, 0.05, 0.70]])
        inv = ordered_index({0: 2, 1: 1, 2: 0})
        assert np.allclose(sorted(stationary(raw)),
                           sorted(stationary(raw[np.ix_(inv, inv)])))


class TestHamiltonFilter:
    """The one-step-ahead predictive score must actually be one-step-ahead.

    The bug being guarded: the old evaluator took the filtered state at the end of training
    and reused it, unchanged, for every observation in the test block — never advancing it
    through P, never updating it after an observation.
    """

    @staticmethod
    def _setup(T=40, seed=0):
        from src.stage4_regime_model import _gaussian_logpdf  # noqa: F401
        rng = np.random.default_rng(seed)
        sigma = np.array([0.2, 0.6])
        mu_t = np.tile(np.array([-3.0, -2.0]), (T, 1))
        y = rng.normal(-3.0, 0.2, T)
        return y, mu_t, sigma

    def test_score_is_a_log_density_not_a_probability(self):
        from src.stage4_regime_model import hamilton_one_step_scores
        y, mu_t, sigma = self._setup()
        s = hamilton_one_step_scores(y, mu_t, sigma, P2, np.array([1.0, 0.0]))
        assert s.shape == (len(y),) and np.isfinite(s).all()

    def test_state_is_updated_between_observations(self):
        """With a static state every score depends only on y_t. With a real filter, changing
        an EARLY observation must change LATER scores."""
        from src.stage4_regime_model import hamilton_one_step_scores
        y, mu_t, sigma = self._setup()
        base = hamilton_one_step_scores(y, mu_t, sigma, P2, np.array([0.5, 0.5]))
        y2 = y.copy()
        y2[0] = -1.0                                  # a strong signal for the wide regime
        moved = hamilton_one_step_scores(y2, mu_t, sigma, P2, np.array([0.5, 0.5]))
        assert not np.allclose(base[5:], moved[5:]), \
            "later scores did not react to an early observation - the state is not updating"

    def test_matches_an_explicit_hand_rolled_recursion(self):
        from src.stage4_regime_model import _gaussian_logpdf, hamilton_one_step_scores
        y, mu_t, sigma = self._setup(T=25, seed=2)
        p = np.array([0.7, 0.3])
        expected = []
        for t in range(len(y)):
            pred = P2 @ p
            dens = np.exp(_gaussian_logpdf(y[t], mu_t[t], sigma))
            expected.append(np.log((pred * dens).sum()))
            p = (pred * dens) / (pred * dens).sum()
        got = hamilton_one_step_scores(y, mu_t, sigma, P2, np.array([0.7, 0.3]))
        assert np.allclose(got, expected, atol=1e-10)

    def test_survives_extreme_outliers_without_underflow(self):
        """log-sum-exp: a 20-sigma observation must give a finite (very negative) score."""
        from src.stage4_regime_model import hamilton_one_step_scores
        y, mu_t, sigma = self._setup(T=10)
        y[5] = -50.0
        s = hamilton_one_step_scores(y, mu_t, sigma, P2, np.array([0.5, 0.5]))
        assert np.isfinite(s).all() and s[5] < s[0]

    def test_better_specified_mean_scores_higher(self):
        from src.stage4_regime_model import hamilton_one_step_scores
        rng = np.random.default_rng(11)
        T = 200
        y = rng.normal(-3.0, 0.2, T)
        sigma = np.array([0.2, 0.6])
        right = np.tile(np.array([-3.0, -2.0]), (T, 1))
        wrong = np.tile(np.array([+1.0, +2.0]), (T, 1))
        p0 = np.array([1.0, 0.0])
        assert (hamilton_one_step_scores(y, right, sigma, P2, p0).mean()
                > hamilton_one_step_scores(y, wrong, sigma, P2, p0).mean())

    def test_rejects_length_mismatch(self):
        from src.stage4_regime_model import hamilton_one_step_scores
        y, mu_t, sigma = self._setup(T=10)
        with pytest.raises(ValueError, match="rows"):
            hamilton_one_step_scores(y[:5], mu_t, sigma, P2, np.array([1.0, 0.0]))


class TestStandardisation:
    """Exog is standardised for numerical stability and un-standardised for reporting.

    The bug being guarded: VIX (sd ~9) alongside 0-1 text constructs (sd ~0.1) is a ~90x
    scale mismatch that makes statsmodels fail with "Could not untransform parameters". The
    fix must not change what a reported coefficient MEANS.
    """

    def test_standardise_produces_zero_mean_unit_sd(self):
        from src.stage4_regime_model import _standardise
        rng = np.random.default_rng(0)
        X = np.column_stack([rng.normal(20, 9, 200), rng.normal(0.5, 0.1, 200)])
        Xs, m, s = _standardise(X)
        assert np.allclose(Xs.mean(axis=0), 0, atol=1e-12)
        assert np.allclose(Xs.std(axis=0), 1, atol=1e-12)
        assert np.allclose(m, X.mean(axis=0)) and np.allclose(s, X.std(axis=0))

    def test_constant_column_does_not_divide_by_zero(self):
        from src.stage4_regime_model import _standardise
        X = np.column_stack([np.ones(50), np.linspace(0, 1, 50)])
        Xs, _, s = _standardise(X)
        assert np.isfinite(Xs).all() and s[0] == 1.0

    def test_unscaling_recovers_the_original_mean_equation(self):
        """const_std + Xs @ beta_std must equal const_orig + X_raw @ beta_orig, exactly."""
        from src.stage4_regime_model import _standardise, unscaled_params

        rng = np.random.default_rng(1)
        X = np.column_stack([rng.normal(20, 9, 100), rng.normal(0.5, 0.1, 100)])
        Xs, m, s = _standardise(X)
        k, n_feat = 2, 2
        const_std = np.array([-3.0, -2.0])
        beta_std = np.array([[0.4, -0.2], [0.1, 0.3]])

        class FakeRes:
            class model:
                param_names = ["const[0]", "const[1]", "sigma2[0]", "sigma2[1]",
                               "x1[0]", "x1[1]", "x2[0]", "x2[1]"]
            params = np.array([const_std[0], const_std[1], 0.04, 0.09,
                               beta_std[0, 0], beta_std[1, 0],
                               beta_std[0, 1], beta_std[1, 1]])
        res = FakeRes()
        res._exog_scaler = (m, s)

        const_o, beta_o, _ = unscaled_params(res, k, n_feat)
        for j in range(k):
            assert np.allclose(const_std[j] + Xs @ beta_std[j],
                               const_o[j] + X @ beta_o[j], atol=1e-9)

    def test_unscaled_beta_is_larger_for_small_scale_features(self):
        """A 0-1 construct has a small sd, so its raw-unit coefficient must be BIGGER than
        its standardised one — the direction that was silently wrong before."""
        from src.stage4_regime_model import _standardise, unscaled_params
        X = np.column_stack([np.linspace(9, 83, 100), np.linspace(0.3, 0.9, 100)])
        _, m, s = _standardise(X)

        class FakeRes:
            class model:
                param_names = ["const[0]", "sigma2[0]", "x1[0]", "x2[0]"]
            params = np.array([-3.0, 0.04, 0.5, 0.5])
        res = FakeRes()
        res._exog_scaler = (m, s)
        _, beta_o, _ = unscaled_params(res, 1, 2)
        assert abs(beta_o[0][1]) > abs(beta_o[0][0]) * 10

    def test_no_scaler_is_a_passthrough(self):
        from src.stage4_regime_model import unscaled_params

        class FakeRes:
            class model:
                param_names = ["const[0]", "sigma2[0]", "x1[0]"]
            params = np.array([-3.0, 0.04, 0.7])
        res = FakeRes()
        res._exog_scaler = None
        const, beta, _ = unscaled_params(res, 1, 1)
        assert const[0] == pytest.approx(-3.0) and beta[0][0] == pytest.approx(0.7)
