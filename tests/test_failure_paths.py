# -*- coding: utf-8 -*-
"""
NEGATIVE-PATH CONTRACTS: the supplied LLM machinery fails CLOSED.

Each test pins one way a failed call, a corrupt or stale cache, or a bad declaration
could previously flow into a reportable-looking artefact:

  - a branch call that failed after retries looked like "the model proposed no channels"
  - a failed evaluator call silently gave every branch credibility 0.5
  - a cached payload was trusted forever because it once said "ok", even when it no
    longer validated against the current response schema
  - an authentication error was retried with exponential backoff it could never fix
  - a duplicated or future-dated sources.json record passed validation
  - a programming error inside shot selection was classified as strategy infeasibility

No test here touches the network: the client is faked and the cache directories are
redirected to a temp folder.
"""
from __future__ import annotations

import json
import os
import sys
import types

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import config  # noqa: E402
import context_docs  # noqa: E402
import decision_replay as dr  # noqa: E402
import scenarios as sc  # noqa: E402
from pydantic import BaseModel as BaseModel_  # noqa: E402
from scenario_engine import Branch  # noqa: E402


# -------------------------------------------------------------------------------------------
# Plumbing: fake clients, fast retries, redirected caches
# -------------------------------------------------------------------------------------------

class AuthenticationError(Exception):
    """Name-matched as non-transient by the retry policy, like openai's class."""


def _response(parsed=None, text=None):
    msg = types.SimpleNamespace(parsed=parsed, content=text)
    # `.request` mirrors what courseapi records: the envelope writers copy it, and the
    # cache's output-ceiling rule reads it back, so a fake without it is not a fake of
    # this adapter.
    return types.SimpleNamespace(
        choices=[types.SimpleNamespace(message=msg)],
        id="fake-request", usage=None,
        request={"model": config.MODEL, "temperature": config.SAMPLING_TEMPERATURE,
                 "max_output_tokens": config.MAX_OUTPUT_TOKENS},
        provenance={"transmitted": {"max_output_tokens": config.MAX_OUTPUT_TOKENS}},
        model=config.MODEL)


def _client(fn):
    """A stand-in OpenAI client whose parse/create both call fn(**kwargs)."""
    comp = types.SimpleNamespace(parse=lambda **kw: fn(**kw), create=lambda **kw: fn(**kw))
    chat = types.SimpleNamespace(completions=comp)
    return types.SimpleNamespace(beta=types.SimpleNamespace(chat=chat), chat=chat)


@pytest.fixture()
def fast_llm(monkeypatch, tmp_path):
    """Two instant retries, caches in a temp dir, and a call counter to install a client."""
    monkeypatch.setattr(config, "MAX_RETRIES", 2)
    monkeypatch.setattr(sc, "RAW_DIR", tmp_path / "shock_raw")
    monkeypatch.setattr(dr, "RAW_DIR", tmp_path / "replay_raw")
    (tmp_path / "shock_raw").mkdir()
    (tmp_path / "replay_raw").mkdir()
    monkeypatch.setattr(sc.time, "sleep", lambda s: None)
    calls = {"n": 0}

    def install(module, fn):
        def counted(**kw):
            calls["n"] += 1
            return fn(**kw)
        monkeypatch.setattr(module, "client_", lambda: _client(counted))
    return types.SimpleNamespace(install=install, calls=calls, tmp=tmp_path)


def _valid_adversarial():
    return sc.AdversarialOut(
        strongest_counterargument="x", channels_you_underweighted=[],
        channel_overweighted="y", what_would_have_to_be_true="z",
        can_construct_case=False)


# -------------------------------------------------------------------------------------------
# Shock: llm_parsed raises, persists failures, and re-validates its cache
# -------------------------------------------------------------------------------------------

def test_llm_parsed_raises_after_transient_retries_and_persists_the_failure(fast_llm):
    def boom(**kw):
        raise TimeoutError("simulated transient failure")
    fast_llm.install(sc, boom)
    with pytest.raises(sc.LLMCallError, match="failed"):
        sc.llm_parsed("s", "u", sc.AdversarialOut)
    assert fast_llm.calls["n"] == config.MAX_RETRIES, "transient errors are retried"
    envelopes = list((fast_llm.tmp / "shock_raw").glob("AdversarialOut-*.json"))
    assert envelopes, "the failure must be persisted as an envelope"
    rec = json.loads(envelopes[0].read_text())
    assert rec["ok"] is False and "TimeoutError" in rec["error"]


def test_llm_parsed_fails_immediately_on_a_non_transient_error(fast_llm):
    def boom(**kw):
        raise AuthenticationError("bad key")
    fast_llm.install(sc, boom)
    with pytest.raises(sc.LLMCallError):
        sc.llm_parsed("s", "u", sc.AdversarialOut)
    assert fast_llm.calls["n"] == 1, "an authentication error must not be retried"


def test_llm_parsed_quarantines_a_cached_payload_that_no_longer_validates(fast_llm):
    fast_llm.install(sc, lambda **kw: _response(parsed=_valid_adversarial()))
    first = sc.llm_parsed("s", "u", sc.AdversarialOut)
    assert first["strongest_counterargument"] == "x"
    assert fast_llm.calls["n"] == 1

    cache = next((fast_llm.tmp / "shock_raw").glob("AdversarialOut-*.json"))
    rec = json.loads(cache.read_text())
    del rec["payload"]["strongest_counterargument"]  # stale schema / corrupt payload
    cache.write_text(json.dumps(rec))

    second = sc.llm_parsed("s", "u", sc.AdversarialOut)
    assert second["strongest_counterargument"] == "x", "refetched, not trusted"
    assert fast_llm.calls["n"] == 2, "the invalid cache must trigger a fresh call"
    assert list((fast_llm.tmp / "shock_raw").glob("*.invalid")), "quarantined, not deleted"


def test_llm_parsed_returns_a_healthy_cache_without_calling(fast_llm):
    fast_llm.install(sc, lambda **kw: _response(parsed=_valid_adversarial()))
    sc.llm_parsed("s", "u", sc.AdversarialOut)
    sc.llm_parsed("s", "u", sc.AdversarialOut)
    assert fast_llm.calls["n"] == 1


def test_evaluate_refuses_to_default_an_unscored_channel(monkeypatch):
    monkeypatch.setattr(sc, "llm_parsed", lambda *a, **k: {"scores": []})
    with pytest.raises(sc.LLMCallError, match="unscored"):
        sc.evaluate([Branch(channel="a channel the evaluator dropped")],
                    "scenario", ["gpr"], "")


def test_evaluate_rejects_an_out_of_range_credibility(monkeypatch):
    monkeypatch.setattr(sc, "llm_parsed", lambda *a, **k: {
        "scores": [{"channel_id": "ch000", "channel": "ch",
                    "credibility": 1.7, "reason": "r"}]})
    with pytest.raises(sc.LLMCallError, match=r"outside \[0, 1\]"):
        sc.evaluate([Branch(channel="ch")], "scenario", ["gpr"], "")


def test_evaluate_rejects_a_duplicate_scored_id(monkeypatch):
    monkeypatch.setattr(sc, "llm_parsed", lambda *a, **k: {
        "scores": [{"channel_id": "ch000", "channel": "a", "credibility": 0.5,
                    "reason": "r"},
                   {"channel_id": "ch000", "channel": "a", "credibility": 0.9,
                    "reason": "r2"}]})
    with pytest.raises(sc.LLMCallError, match="duplicate ids"):
        sc.evaluate([Branch(channel="a"), Branch(channel="b")], "s", ["gpr"], "")


def test_evaluate_rejects_an_unknown_scored_id(monkeypatch):
    monkeypatch.setattr(sc, "llm_parsed", lambda *a, **k: {
        "scores": [{"channel_id": "ch999", "channel": "??", "credibility": 0.5,
                    "reason": "r"}]})
    with pytest.raises(sc.LLMCallError, match="unknown ids"):
        sc.evaluate([Branch(channel="a")], "s", ["gpr"], "")


def test_evaluate_cannot_collide_two_similarly_named_channels(monkeypatch):
    """Name-prefix matching once applied one score to two 60-char-identical channels."""
    long_a = "Rising import prices from increased shipping costs " + "x" * 30 + " A"
    long_b = "Rising import prices from increased shipping costs " + "x" * 30 + " B"
    scores = {"ch000": 0.9, "ch001": 0.1}

    def fake(sys_p, user, schema, seed_offset=0):
        import json as _json
        sent = _json.loads(user.split("CHANNELS:\n", 1)[1])
        return {"scores": [{"channel_id": p["id"], "channel": p["channel"],
                            "credibility": scores[p["id"]], "reason": "r"}
                           for p in sent]}
    monkeypatch.setattr(sc, "llm_parsed", fake)
    a, b = Branch(channel=long_a), Branch(channel=long_b)
    sc.evaluate([a, b], "s", ["gpr"], "")
    assert (a.score, b.score) == (0.9, 0.1), "each channel must get ITS OWN score"


# -------------------------------------------------------------------------------------------
# Replay: the cached-call layer and the reportability guards
# -------------------------------------------------------------------------------------------

def test_cached_call_quarantines_an_empty_text_payload(fast_llm):
    key = dr._cache_key("stmt", "s", "u", 0)
    path = dr.RAW_DIR / f"stmt-{key}.json"
    path.write_text(json.dumps({"ok": True, "kind": "stmt", "payload": {"text": "  "}}))
    fast_llm.install(dr, lambda **kw: _response(text="a real statement"))
    rec = dr._cached_call("stmt", "s", "u")
    assert rec["ok"] and rec["payload"]["text"] == "a real statement"
    assert fast_llm.calls["n"] == 1
    assert list(dr.RAW_DIR.glob("*.invalid"))


def test_cached_call_revalidates_schema_payloads_on_load(fast_llm):
    key = dr._cache_key("audit", dr.AUDIT_PROMPT, "u", 0, schema=dr.ClaimAudit)
    path = dr.RAW_DIR / f"audit-{key}.json"
    path.write_text(json.dumps({"ok": True, "kind": "audit", "payload": {
        "claims": [{"claim": "x", "category": "not-a-real-category"}]}}))
    good = dr.ClaimAudit(claims=[{"claim": "x", "category": "unverifiable"}])
    fast_llm.install(dr, lambda **kw: _response(parsed=good))
    rec = dr._cached_call("audit", dr.AUDIT_PROMPT, "u", schema=dr.ClaimAudit)
    assert rec["ok"] and rec["payload"]["claims"][0]["category"] == "unverifiable"
    assert fast_llm.calls["n"] == 1, "the stale cache must be refetched"
    assert list(dr.RAW_DIR.glob("*.invalid"))


def test_cached_call_fails_fast_on_a_non_transient_error(fast_llm):
    def boom(**kw):
        raise AuthenticationError("bad key")
    fast_llm.install(dr, boom)
    with pytest.raises(dr.LLMCallError, match="non-transient"):
        dr._cached_call("rec", "s", "u")
    assert fast_llm.calls["n"] == 1


def test_generate_statement_blocks_on_a_failed_call(monkeypatch):
    monkeypatch.setattr(dr, "_cached_call",
                        lambda *a, **k: {"ok": False, "error": "boom"})
    with pytest.raises(dr.LLMCallError, match="no statement"):
        dr.generate_statement({"recommendation": "hold"}, {})


def test_generate_statement_blocks_on_empty_text(monkeypatch):
    monkeypatch.setattr(dr, "_cached_call",
                        lambda *a, **k: {"ok": True, "payload": {"text": "   "}})
    with pytest.raises(dr.LLMCallError, match="empty"):
        dr.generate_statement({"recommendation": "hold"}, {})


def test_audit_statement_blocks_on_a_failed_call(monkeypatch):
    monkeypatch.setattr(dr, "_cached_call",
                        lambda *a, **k: {"ok": False, "error": "boom"})
    with pytest.raises(dr.LLMCallError, match="unaudited"):
        dr.audit_statement("a statement", {})


# -------------------------------------------------------------------------------------------
# Feasibility: only nshot's ValueError is infeasibility; anything else is a defect
# -------------------------------------------------------------------------------------------

def test_feasibility_propagates_unexpected_exceptions(monkeypatch):
    import pandas as pd
    monkeypatch.setattr(dr, "_panel", lambda *a, **k: pd.DataFrame())

    def broken(*a, **k):
        raise KeyError("slope_cash3y")  # a missing column is a defect, not infeasibility
    monkeypatch.setattr(dr.nshot, "select_shots", broken)
    with pytest.raises(KeyError):
        dr.feasible_strategies("2013-08-06")
    with pytest.raises(KeyError):
        dr.feasible_everywhere(["2013-08-06"], list(dr.nshot.STRATEGIES), 8)


def test_feasibility_still_classifies_valueerror_as_infeasible(monkeypatch):
    import pandas as pd
    monkeypatch.setattr(dr, "_panel", lambda *a, **k: pd.DataFrame())

    def infeasible(*a, **k):
        raise ValueError("k=8 but only 3 eligible meetings precede the target")
    monkeypatch.setattr(dr.nshot, "select_shots", infeasible)
    assert dr.feasible_strategies("2013-08-06") == []
    keep, dropped = dr.feasible_everywhere(["2013-08-06"], list(dr.nshot.STRATEGIES), 8)
    assert keep == [] and dropped


# -------------------------------------------------------------------------------------------
# sources.json: ordinary mistakes must fail loudly
# -------------------------------------------------------------------------------------------

def _write_sources(tmp_path, records):
    (tmp_path / "sources.json").write_text(json.dumps(records), encoding="utf-8")
    return tmp_path


GOOD = {"file": "a.pdf", "organisation": "IMF", "title": "WEO",
        "url": "https://imf.org/weo", "retrieved": "2026-08-12"}


def test_sources_rejects_a_duplicate_filename(tmp_path):
    d = _write_sources(tmp_path, [GOOD, {**GOOD, "title": "WEO second copy"}])
    with pytest.raises(RuntimeError, match="duplicate declaration"):
        context_docs._declared_sources(d)


def test_sources_rejects_a_future_retrieval_date(tmp_path):
    d = _write_sources(tmp_path, [{**GOOD, "retrieved": "2999-01-01"}])
    with pytest.raises(RuntimeError, match="future"):
        context_docs._declared_sources(d)


def test_sources_rejects_one_document_declared_under_two_filenames(tmp_path):
    d = _write_sources(tmp_path, [GOOD, {**GOOD, "file": "b.pdf"}])
    with pytest.raises(RuntimeError, match="same organisation and title"):
        context_docs._declared_sources(d)


def test_sources_accepts_a_clean_declaration(tmp_path):
    other = {"file": "b.pdf", "organisation": "WEF", "title": "Global Risks",
             "url": "https://weforum.org/risks", "retrieved": "2026-08-12"}
    d = _write_sources(tmp_path, [GOOD, other])
    out = context_docs._declared_sources(d)
    assert set(out) == {"a.pdf", "b.pdf"}


# -------------------------------------------------------------------------------------------
# Raw structured inputs: ordinary corruption must fail loudly, before pandas or sklearn do
# -------------------------------------------------------------------------------------------

def _rates_frame(rows, **overrides):
    import numpy as np
    import pandas as pd
    idx = pd.date_range("2006-10-02", periods=rows, freq="B")
    cols = {"FIRMMCRTD": 4.0, "FIRMMOIS1D": 4.1, "FIRMMOIS3D": 4.2,
            "FIRMMOIS6D": 4.3, "FIRMMBAB90D": 4.4,
            "FCMYGBAG3D": 4.5, "FCMYGBAG10D": 4.8}
    data = {c: np.full(rows, v) for c, v in cols.items()}
    data.update(overrides)
    return pd.DataFrame(data, index=idx)


def test_zero_row_rate_files_fail_the_input_contract(monkeypatch):
    import data_rates
    empty = _rates_frame(0)
    monkeypatch.setattr(data_rates, "_read_rba_excel", lambda *a, **k: empty)
    monkeypatch.setattr(data_rates, "_read_rba_csv", lambda *a, **k: empty)
    with pytest.raises(RuntimeError, match="required rate series failed to build"):
        data_rates.load_rates_daily()


def test_non_finite_rate_values_fail_the_input_contract(monkeypatch):
    import numpy as np
    import data_rates
    bad = np.full(5000, 4.4)
    bad[100] = np.inf
    frame = _rates_frame(5000, FIRMMBAB90D=bad)
    monkeypatch.setattr(data_rates, "_read_rba_excel", lambda *a, **k: frame)
    monkeypatch.setattr(data_rates, "_read_rba_csv", lambda *a, **k: frame)
    with pytest.raises(RuntimeError, match="non-finite|plausible range"):
        data_rates.load_rates_daily()


def test_a_decision_series_with_no_cuts_fails_the_input_contract():
    import pandas as pd
    import data_rates
    idx = pd.to_datetime(["2007-03-06", "2010-11-02", "2024-06-18"])
    dec = pd.DataFrame({"change_pct": [0.25, 0.25, 0.25],
                        "new_target": [6.0, 4.75, 4.35]}, index=idx)
    with pytest.raises(RuntimeError, match="no cuts"):
        data_rates._check_decisions(dec)
    with pytest.raises(RuntimeError, match="no decisions"):
        data_rates._check_decisions(dec.iloc[0:0])


def test_atomic_write_replaces_not_truncates(tmp_path):
    """A crash mid-write must leave the previous artefact intact, never a partial file."""
    target = tmp_path / "artefact.json"
    target.write_text('{"old": true}')
    import config as cfg
    cfg.atomic_write_text(target, '{"new": true}')
    assert json.loads(target.read_text()) == {"new": True}
    # the temp sibling never lingers
    assert not list(tmp_path.glob("*.tmp-write"))


def test_atomic_write_crash_before_replace_leaves_the_old_artefact(tmp_path,
                                                                   monkeypatch):
    """FAULT INJECTION: os.replace() dies. The previous artefact must remain readable
    byte-for-byte, the temp file must be cleaned up, and the failure must propagate -
    so a stage can never mark itself complete over a half-written output."""
    import os as _os
    import config as cfg
    target = tmp_path / "artefact.json"
    target.write_text('{"old": true}')

    def boom(src, dst):
        raise OSError("simulated crash during replace")
    monkeypatch.setattr(_os, "replace", boom)
    with pytest.raises(OSError, match="simulated crash"):
        cfg.atomic_write_text(target, '{"new": true}')
    assert json.loads(target.read_text()) == {"old": True}, "old artefact must survive"
    assert not list(tmp_path.glob("*.tmp-write")), "the temp file must be cleaned up"


def test_atomic_parquet_crash_before_replace_leaves_the_old_artefact(tmp_path,
                                                                     monkeypatch):
    import os as _os
    import pandas as pd
    import config as cfg
    target = tmp_path / "scores.parquet"
    pd.DataFrame({"a": [1, 2]}).to_parquet(target, index=False)

    def boom(src, dst):
        raise OSError("simulated crash during replace")
    monkeypatch.setattr(_os, "replace", boom)
    with pytest.raises(OSError, match="simulated crash"):
        cfg.atomic_to_parquet(pd.DataFrame({"a": [9]}), target)
    assert pd.read_parquet(target)["a"].tolist() == [1, 2], "old artefact must survive"
    assert not list(tmp_path.glob("*.tmp-write"))


# -------------------------------------------------------------------------------------------
# Chronological coverage, PER COLUMN: a frame whose index spans the corpus while one
# required feature or series holds only a sliver must fail
# -------------------------------------------------------------------------------------------

def _market_frame():
    import numpy as np
    import pandas as pd
    import data_panel as dp
    idx = pd.date_range(config.CORPUS_START,
                        pd.Timestamp(config.CORPUS_LAST_MEETING)
                        + pd.Timedelta(days=10), freq="B")
    return pd.DataFrame({c: np.full(len(idx), 0.01) for c in dp.MARKET_FEATURES},
                        index=idx)


def test_a_market_feature_with_only_its_final_values_fails_per_feature_coverage():
    """A 2006-2026 file once passed while rv_5 held only its final twenty values."""
    import numpy as np
    import data_panel as dp
    m = _market_frame()
    m.loc[m.index[:-20], "rv_5"] = np.nan
    with pytest.raises(RuntimeError, match="rv_5"):
        dp._check_market(m)


def test_a_market_feature_ending_before_the_last_meeting_fails():
    import numpy as np
    import data_panel as dp
    m = _market_frame()
    m.loc[m.index > "2020-01-01", "rv_63"] = np.nan
    with pytest.raises(RuntimeError, match="rv_63 ends"):
        dp._check_market(m)


def test_an_intact_market_frame_passes_including_legitimate_warmup_gaps():
    import numpy as np
    import data_panel as dp
    m = _market_frame()
    m.loc[m.index[:63], "rv_63"] = np.nan  # a rolling window warming up is not a gap
    dp._check_market(m)


def _macro_frames(sent_start: str, union_end: str) -> dict:
    import pandas as pd
    import data_macro as dm
    ends = (pd.date_range("2006-01-01", "2026-06-01", freq="MS")
            + pd.offsets.MonthEnd(0))
    frames = {}
    for fname, mapping in dm.SERIES.items():
        cols = {}
        for sid, (name, _lag) in mapping.items():
            s = pd.Series(2.5, index=ends)
            if name == "consumer_sentiment":
                s = s[s.index >= pd.Timestamp(sent_start)]
            if name == "infexp_union_1y":
                s = s[s.index <= pd.Timestamp(union_end)]
            cols[sid] = s
        frames[fname] = pd.DataFrame(cols)
    return frames


def _fake_gpr():
    import pandas as pd
    import data_macro as dm
    ends = (pd.date_range("2006-01-01", "2026-06-01", freq="MS")
            + pd.offsets.MonthEnd(0))
    return pd.concat([pd.DataFrame({"period_end": ends, "series": name,
                                    "value": 100.0,
                                    "published_on": ends + pd.Timedelta(days=30)})
                      for name in dm.GPR_SERIES.values()], ignore_index=True)


def test_macro_exemptions_are_bounded_at_their_documented_endpoints(monkeypatch):
    """The two named exemptions must not accept ANY truncation: consumer_sentiment must
    still start by its documented ~2010-03 onset and infexp_union_1y must still reach
    its documented ~2023-06 discontinuation window."""
    import data_macro as dm

    def run_with(sent_start, union_end):
        frames = _macro_frames(sent_start, union_end)
        monkeypatch.setattr(dm, "_read_rba_csv", lambda f: frames[f])
        monkeypatch.setattr(dm, "load_gpr", _fake_gpr)
        return dm.load_series()

    run_with("2010-01-01", "2023-09-30")  # the documented reality passes
    with pytest.raises(RuntimeError, match="onset"):
        run_with("2024-01-01", "2023-09-30")
    with pytest.raises(RuntimeError, match="discontinuation"):
        run_with("2010-01-01", "2009-12-31")


def test_a_truncated_meeting_calendar_fails_the_contract(monkeypatch, tmp_path):
    """200 meetings ending mid-2023 once passed a count-free calendar check."""
    import pandas as pd
    import data_panel as dp
    cal = tmp_path / "meeting_calendar.csv"
    pd.DataFrame({"meeting_date": pd.date_range("2006-10-01", periods=200,
                                                freq="30D")}).to_csv(cal, index=False)
    monkeypatch.setattr(config, "MEETING_CALENDAR", cal)
    with pytest.raises(RuntimeError, match="meeting calendar failed"):
        dp.meeting_calendar()


# -------------------------------------------------------------------------------------------
# THE shared construct-score contract, on every consumer
# -------------------------------------------------------------------------------------------

def _calendar_dates():
    import pandas as pd
    return pd.to_datetime(pd.read_csv(config.MEETING_CALENDAR)["meeting_date"])


def _scores_frame(dates):
    import pandas as pd
    d = pd.DataFrame({"meeting_date": pd.to_datetime(pd.Series(list(dates)))})
    d["n_calls_valid"] = config.N_PARALLEL_CALLS
    for c in config.TEXT_FEATURES:
        d[c] = 0.5
        d[f"{c}_sd"] = 0.1
    return d


def test_construct_scores_without_a_meeting_date_column_fail_by_name():
    """This used to surface as a bare KeyError from deep inside pandas."""
    import pandas as pd
    with pytest.raises(RuntimeError, match="meeting_date"):
        config.validate_construct_scores(pd.DataFrame({"policy_stance": [0.5]}))


def test_one_scored_meeting_passes_development_but_fails_a_reportable_run():
    df = _scores_frame(_calendar_dates().iloc[:1])
    config.validate_construct_scores(df, reportable=False)
    with pytest.raises(RuntimeError, match="EXACTLY"):
        config.validate_construct_scores(df, reportable=True)


def test_both_reportable_entry_points_reject_a_partially_scored_corpus(monkeypatch,
                                                                       tmp_path):
    """Replay's and Shock's guards must be the SAME contract, not two row counts."""
    df = _scores_frame(_calendar_dates().iloc[:-1])
    p = tmp_path / "construct_scores.parquet"
    df.to_parquet(p, index=False)
    monkeypatch.setattr(config, "CONSTRUCT_SCORES", p)
    with pytest.raises(RuntimeError, match="reportable"):
        dr._require_complete_scores()
    with pytest.raises(RuntimeError, match="reportable"):
        sc._require_complete_scores()


def test_scores_with_too_few_valid_calls_or_a_negative_sd_fail():
    df = _scores_frame(_calendar_dates())
    df.loc[0, "n_calls_valid"] = config.MIN_VALID_CALLS - 1
    with pytest.raises(RuntimeError, match="n_calls_valid"):
        config.validate_construct_scores(df)
    df2 = _scores_frame(_calendar_dates())
    df2.loc[0, "policy_stance_sd"] = -0.2
    with pytest.raises(RuntimeError, match="sd columns"):
        config.validate_construct_scores(df2)


def test_an_empty_score_table_or_missing_sd_columns_fail():
    """An existing zero-row table is corruption, and six missing sd columns once passed
    because only the sd columns that HAPPENED to exist were checked."""
    df = _scores_frame(_calendar_dates())
    with pytest.raises(RuntimeError, match="EMPTY"):
        config.validate_construct_scores(df.iloc[0:0])
    df2 = df.drop(columns=[f"{c}_sd" for c in config.TEXT_FEATURES[1:]])
    with pytest.raises(RuntimeError, match="missing sd columns"):
        config.validate_construct_scores(df2)


def test_nan_or_fractional_call_counts_fail():
    import numpy as np
    df = _scores_frame(_calendar_dates())
    df["n_calls_valid"] = df["n_calls_valid"].astype(float)
    df.loc[0, "n_calls_valid"] = np.nan
    with pytest.raises(RuntimeError, match="whole number"):
        config.validate_construct_scores(df)
    df2 = _scores_frame(_calendar_dates())
    df2["n_calls_valid"] = df2["n_calls_valid"].astype(float)
    df2.loc[0, "n_calls_valid"] = 3.5
    with pytest.raises(RuntimeError, match="whole number"):
        config.validate_construct_scores(df2)


def test_team_construct_scores_are_what_the_reaction_profiles_rank(monkeypatch,
                                                                   tmp_path):
    """WORDS MUST FEED SHOCK: the episode profiles rank the TEAM's validated scores
    overlaid on the frozen panel, and changing one team score changes both the profile
    input and the Shock input fingerprint."""
    import pandas as pd
    dates = _calendar_dates()
    a = _scores_frame(dates)
    b = _scores_frame(dates)
    b.loc[0, "policy_stance"] = 0.9
    pa, pb = tmp_path / "a.parquet", tmp_path / "b.parquet"
    a.to_parquet(pa, index=False)
    b.to_parquet(pb, index=False)
    panel = pd.DataFrame(
        {c: 0.123 for c in config.TEXT_FEATURES} | {"cash_rate": 4.0},
        index=pd.DatetimeIndex(pd.to_datetime(dates), name="meeting_date"))

    monkeypatch.setattr(config, "CONSTRUCT_SCORES", pa)
    out_a = sc._profile_panel(panel)
    assert (out_a[config.TEXT_FEATURES[0]].iloc[0]
            == a[config.TEXT_FEATURES[0]].iloc[0]), (
        "the profile panel must carry the TEAM's scores, not the frozen panel's")
    fp_a = sc.input_fingerprints()["construct_scores"]

    monkeypatch.setattr(config, "CONSTRUCT_SCORES", pb)
    out_b = sc._profile_panel(panel)
    assert out_b["policy_stance"].iloc[0] == pytest.approx(0.9)
    fp_b = sc.input_fingerprints()["construct_scores"]
    assert fp_a != fp_b, (
        "changing a team construct score must change the Shock input fingerprint")


# -------------------------------------------------------------------------------------------
# Envelope ledgers: the tolerated-failure path, exercised directly
# -------------------------------------------------------------------------------------------

def test_a_tolerated_benchmark_failure_is_ledgered_with_its_failed_state(fast_llm,
                                                                         monkeypatch):
    """The worked ledger holds zero failed calls, so this branch never runs on real
    artefacts - it is pinned here instead."""
    monkeypatch.setattr(dr.time, "sleep", lambda s: None)

    def boom(**kw):
        raise TimeoutError("simulated transient failure")
    fast_llm.install(dr, boom)
    config.ledger_reset("replay")
    rec = dr._cached_call("rec", "s", "u")
    assert rec["ok"] is False
    ledger = config._ENVELOPE_LEDGERS["replay"]
    assert list(ledger.values()) == [False], (
        "the failed benchmark call must be ledgered with its failed state")
    config.ledger_reset("replay")


def test_a_matching_failed_benchmark_envelope_is_accepted(tmp_path):
    import hashlib
    f = tmp_path / "rec-abc.json"
    f.write_text(json.dumps({"ok": False, "error": "boom"}))
    entry = {"path": "rec-abc.json", "ok": False,
             "sha256": hashlib.sha256(f.read_bytes()).hexdigest()}
    config.validate_envelope_entry("replay", entry, root=tmp_path)  # must not raise


def test_a_failed_envelope_outside_replay_benchmarks_is_rejected(tmp_path):
    """Failures are bounded to Replay's rec-* attrition: a failed statement or audit
    call, or ANY failed Shock call, must fail the submission even when ledgered."""
    import hashlib
    for stage, name in (("replay", "stmt-abc.json"), ("shock", "rec-abc.json")):
        f = tmp_path / name
        f.write_text(json.dumps({"ok": False, "error": "boom"}))
        entry = {"path": name, "ok": False,
                 "sha256": hashlib.sha256(f.read_bytes()).hexdigest()}
        with pytest.raises(RuntimeError, match="does not tolerate failure"):
            config.validate_envelope_entry(stage, entry, root=tmp_path)


def test_an_envelope_whose_state_or_bytes_drifted_is_rejected(tmp_path):
    import hashlib
    f = tmp_path / "stmt-abc.json"
    f.write_text(json.dumps({"ok": False, "error": "boom"}))
    entry = {"path": "stmt-abc.json", "ok": True,
             "sha256": hashlib.sha256(f.read_bytes()).hexdigest()}
    with pytest.raises(RuntimeError, match="success state no longer matches"):
        config.validate_envelope_entry("replay", entry, root=tmp_path)
    f.write_text(json.dumps({"ok": True, "payload": {"text": "hi"}}))
    with pytest.raises(RuntimeError, match="changed since"):
        config.validate_envelope_entry("replay", entry, root=tmp_path)


# -------------------------------------------------------------------------------------------
# THE MIGRATION REVIEW, 12 September 2026. One test per defect it found, so a regression
# shows up as a named failure rather than as a green suite.
# -------------------------------------------------------------------------------------------

class _Quota(Exception):
    """Stands in for unsw_ai.DailyQuotaExceededError, matched by NAME like the real one."""


_Quota.__name__ = "DailyQuotaExceededError"


def _fake_parsed(tf):
    schema = tf._schema_model()
    return {f: ("a verbatim quote" if f.endswith("_evidence") else 50)
            for f in schema.model_fields}


def _draw(tf, idx, ok=True, error=None):
    rec = {"call_index": idx, "config_hash": tf.call_config_hash(),
           "request": {"model": config.MODEL,
                       "temperature": config.SAMPLING_TEMPERATURE,
                       "max_output_tokens": config.MAX_OUTPUT_TOKENS},
           "timestamp": "2026-09-12T00:00:00+00:00"}
    if ok:
        rec.update({"ok": True, "parsed": _fake_parsed(tf), "model": config.MODEL,
                    "temperature": config.SAMPLING_TEMPERATURE, "attempt": 1})
    else:
        rec.update({"ok": False, "error": error or "ContentFilteredError: blocked"})
    return rec


@pytest.fixture()
def words(tmp_path, monkeypatch):
    """text_features with its cache directory redirected and the abort flag cleared."""
    import text_features as tf
    monkeypatch.setattr(tf, "run_dir", lambda: tmp_path)
    tf._RUN_ABORTED.clear()
    config.ledger_reset("words")
    return tf


def test_successful_draws_survive_a_quota_failure_mid_document(words, monkeypatch):
    """
    THE GAP THIS CLOSES. The document cache was written only after EVERY future returned,
    so one DailyQuotaExceededError on the fifth call discarded the four that had already
    succeeded and been paid for. A probe found zero saved envelopes after four successes.
    """
    calls = {"n": 0}

    def four_then_quota(text, call_index):
        calls["n"] += 1
        if calls["n"] >= config.N_PARALLEL_CALLS:
            raise _Quota("daily budget exhausted")
        return _draw(words, call_index)

    monkeypatch.setattr(words, "score_once", four_then_quota)
    with pytest.raises(_Quota):
        words.score_document("2020-01-01", "the minutes")
    saved = json.loads((words.run_dir() / "2020-01-01.json").read_text())
    assert sum(1 for c in saved if c.get("ok")) == config.N_PARALLEL_CALLS - 1, (
        "every call that completed before the quota failure must be on disk")


def test_a_partial_cache_requests_only_the_missing_draws(words, monkeypatch):
    """A document holding four valid draws must cost ONE call, not five."""
    have = [_draw(words, config.CALL_INDEX_BASE + i) for i in range(4)]
    (words.run_dir() / "2020-01-01.json").write_text(json.dumps(have))
    calls = {"n": 0}

    def counted(text, call_index):
        calls["n"] += 1
        return _draw(words, call_index)

    monkeypatch.setattr(words, "score_once", counted)
    rec = words.score_document("2020-01-01", "the minutes")
    assert calls["n"] == 1, f"expected 1 fresh call, made {calls['n']}"
    assert len(rec["calls"]) == config.N_PARALLEL_CALLS


def test_settled_filter_failures_are_not_retried_every_run(words, monkeypatch):
    """
    Three valid draws and two content-filtered ones meets MIN_VALID_CALLS. Re-calling the
    filtered indices spends the shared budget to reproduce the same refusal, so a document
    in that state is complete.
    """
    have = [_draw(words, config.CALL_INDEX_BASE + i) for i in range(3)]
    have += [_draw(words, config.CALL_INDEX_BASE + i, ok=False) for i in (3, 4)]
    (words.run_dir() / "2021-02-02.json").write_text(json.dumps(have))

    def boom(text, call_index):
        raise AssertionError("a settled failure must not be re-called")

    monkeypatch.setattr(words, "score_once", boom)
    rec = words.score_document("2021-02-02", "the minutes")
    assert rec["cached"] is True


def test_a_transient_failure_IS_retried(words, monkeypatch):
    """The converse: a timeout is not settled, so the draw is requested again."""
    have = [_draw(words, config.CALL_INDEX_BASE + i) for i in range(4)]
    have.append(_draw(words, config.CALL_INDEX_BASE + 4, ok=False,
                      error="TimeoutError: connection dropped"))
    (words.run_dir() / "2021-03-03.json").write_text(json.dumps(have))
    calls = {"n": 0}

    def counted(text, call_index):
        calls["n"] += 1
        return _draw(words, call_index)

    monkeypatch.setattr(words, "score_once", counted)
    words.score_document("2021-03-03", "the minutes")
    assert calls["n"] == 1, "a transient failure must be retried, unlike a settled one"


def test_offline_mode_never_calls_and_refuses_an_under_covered_document(words, monkeypatch):
    def boom(text, call_index):
        raise AssertionError("--offline must not contact the service")

    monkeypatch.setattr(words, "score_once", boom)
    enough = [_draw(words, config.CALL_INDEX_BASE + i) for i in range(3)]
    (words.run_dir() / "2022-01-01.json").write_text(json.dumps(enough))
    rec = words.score_document("2022-01-01", "the minutes", offline=True)
    assert rec["cached"] is True

    thin = [_draw(words, config.CALL_INDEX_BASE + i) for i in range(2)]
    (words.run_dir() / "2022-02-02.json").write_text(json.dumps(thin))
    with pytest.raises(RuntimeError, match="offline"):
        words.score_document("2022-02-02", "the minutes", offline=True)


# -------------------------------------------------------------------------------------------
# The API adapter: incomplete responses, and parameters that were silently dropped
# -------------------------------------------------------------------------------------------

class _Details:
    def __init__(self, reason):
        self.reason = reason


class _Raw:
    """A Responses-shaped object with a settable status."""

    def __init__(self, status="completed", reason=None, text="an answer"):
        self.status = status
        self.incomplete_details = _Details(reason) if reason else None
        self.output_text = text
        self.output = []
        self.model = "gpt-5.4-mini"
        self.usage = None
        self.id = "resp_test"


def test_a_content_filtered_completion_is_not_a_successful_call():
    """
    THE GAP THIS CLOSES. A 200 with status="incomplete", reason="content_filter" and
    partial text was accepted as an ordinary completed response, given finish_reason
    "stop", and cached by Replay as ok=True.
    """
    import courseapi
    import unsw_ai
    with pytest.raises(unsw_ai.ContentFilteredError):
        courseapi.validate_response(_Raw("incomplete", "content_filter", "half an ans"))


def test_a_truncated_answer_is_not_a_successful_call():
    import courseapi
    with pytest.raises(courseapi.IncompleteResponseError, match="output ceiling"):
        courseapi.validate_response(_Raw("incomplete", "max_output_tokens", "cut off mid"))


def test_a_completed_response_passes_validation():
    import courseapi
    courseapi.validate_response(_Raw("completed"))          # must not raise
    courseapi.validate_response(_Raw(None))                 # nor an object without status


def test_truncation_and_filtering_are_non_transient_for_every_stage():
    """Retrying either at the same settings spends budget to get the same refusal."""
    import courseapi
    import decision_replay as dr
    import scenarios as sc
    import text_features as tf
    for name in ("IncompleteResponseError", "ContentFilteredError"):
        assert name in courseapi.NON_TRANSIENT_ERRORS
        for mod in (tf, dr, sc):
            assert name in mod._NON_TRANSIENT, f"{mod.__name__} would retry {name}"


@pytest.mark.parametrize("param", ["seed", "reasoning", "n", "stop", "logit_bias"])
def test_the_adapter_rejects_parameters_it_would_not_transmit(param):
    """
    A probe passed `seed` and `reasoning`; neither was rejected and neither was sent. A
    silently dropped parameter is a false entry in the reproducibility record.
    """
    import courseapi
    import unsw_ai
    with pytest.raises(unsw_ai.ParameterNotSupportedError, match=param):
        courseapi.validate_call_kwargs({param: "whatever"})


def test_the_transmitted_record_names_the_output_ceiling():
    """The ceiling changes the answer, so it must appear in what the envelope records."""
    import courseapi
    req = courseapi._Completions._transmitted(None, 1.0, None)
    assert req["max_output_tokens"] == config.MAX_OUTPUT_TOKENS
    assert req["model"] == config.MODEL


def test_the_output_ceiling_is_inside_every_stage_hash():
    """Lowering it truncates answers, so it must invalidate the committed artefacts."""
    import decision_replay as dr
    import scenarios as sc
    import text_features as tf
    for mod in (tf, dr, sc):
        before = mod.config_hash()
        old = config.MAX_OUTPUT_TOKENS
        try:
            config.MAX_OUTPUT_TOKENS = old // 2
            assert mod.config_hash() != before, (
                f"{mod.__name__}: halving the output ceiling left the stage hash "
                f"unchanged, so a truncating configuration would pass as current")
        finally:
            config.MAX_OUTPUT_TOKENS = old


# -------------------------------------------------------------------------------------------
# The label embargo is no longer optional
# -------------------------------------------------------------------------------------------

def _tiny_design(n=120):
    import numpy as np
    import pandas as pd
    idx = pd.date_range("2006-01-01", periods=n, freq="30D")
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)}, index=idx)
    y = pd.Series(rng.integers(0, 3, size=n), index=idx)
    return X, y


def test_rolling_origin_refuses_to_run_without_label_availability():
    """
    THE GAP THIS CLOSES. `available_on=None` was the DEFAULT, so a team could follow the
    instruction to use the supplied evaluator and still train on labels that had not
    resolved - the exact leak the assignment deducts 20 points for.
    """
    import evaluation as ev
    X, y = _tiny_design()
    with pytest.raises(ValueError, match="available_on"):
        ev.rolling_origin(X, y)


def test_the_leaky_arm_is_still_available_but_has_to_be_asked_for():
    import evaluation as ev
    X, y = _tiny_design()
    preds = ev.rolling_origin(X, y, available_on=None, allow_unresolved_labels=True)
    assert len(preds), "the deliberate teaching comparison must still work"
    assert not preds["label_embargo"].any(), (
        "rows produced without the embargo must be stamped as such")


def test_embargoed_rows_are_stamped_as_embargoed():
    import pandas as pd
    import evaluation as ev
    X, y = _tiny_design()
    avail = pd.Series(y.index + pd.Timedelta(days=182), index=y.index)
    preds = ev.rolling_origin(X, y, available_on=avail)
    if len(preds):
        assert preds["label_embargo"].all()


def test_staleness_cost_will_not_run_two_leaky_arms():
    """
    THE GAP THIS CLOSES. This supplied helper called rolling_origin TWICE with no
    availability dates, so a team following the instruction to use the supplied evaluator
    produced two leaky numbers and a difference between them - and both arms leaked the
    same way, which is the kind of error that survives a sanity check.
    """
    import attribution as at
    import pandas as pd
    n = 50
    panel = pd.DataFrame(
        {"a": [1.0] * n,
         "a__age_days": [70.0] * n,                      # staleness_table needs this
         "y_cycle": ([0, 1, 2] * n)[:n]},
        index=pd.date_range("2010-01-01", periods=n, freq="30D"))
    with pytest.raises(ValueError, match="label-availability"):
        at.staleness_cost(panel, {"macro": ["a"]}, "y_cycle", [0, 1, 2], ["macro"])

class _Schema(BaseModel_):
    ok: bool = True


def _fake_client(raw, *, structured=False, record=None):
    """A stand-in UNSWInstructor whose raw responses.create returns `raw`."""
    import types

    def responses_create(**kw):
        if record is not None:
            record.append(kw)
        return raw

    def create_with_completion(**kw):
        if record is not None:
            record.append(kw)
        return types.SimpleNamespace(ok=True), raw

    return types.SimpleNamespace(
        client=types.SimpleNamespace(
            responses=types.SimpleNamespace(create=responses_create)),
        chat=types.SimpleNamespace(completions=types.SimpleNamespace(
            create_with_completion=create_with_completion)),
        settings=types.SimpleNamespace(fallback_models=(), proxy_url="https://x",
                                       student_id="9999999", model=config.MODEL))


@pytest.mark.parametrize("path", ["free_text", "structured"])
def test_BOTH_call_paths_reject_an_incomplete_response(monkeypatch, path):
    """
    THE GAP THIS CLOSES. `create()` went straight to the SDK's responses.create,
    skipping the wrapper's error translation, its 429 backoff and every status check,
    while `parse()` went through instructor. A filtered or truncated completion was
    therefore a success on one path and an error on the other.
    """
    import courseapi
    import unsw_ai
    raw = _Raw("incomplete", "content_filter", "half an answer")
    monkeypatch.setattr(unsw_ai, "get_client", lambda *a, **k: _fake_client(raw))
    comp = courseapi._Completions(structured=(path == "structured"))
    msgs = [{"role": "user", "content": "hello"}]
    with pytest.raises(unsw_ai.ContentFilteredError):
        if path == "free_text":
            comp.create(messages=msgs, temperature=1.0)
        else:
            comp.parse(messages=msgs, response_format=_Schema, temperature=1.0)


@pytest.mark.parametrize("path", ["free_text", "structured"])
def test_BOTH_call_paths_transmit_the_output_ceiling(monkeypatch, path):
    import courseapi
    import unsw_ai
    sent = []
    raw = _Raw("completed")
    monkeypatch.setattr(unsw_ai, "get_client",
                        lambda *a, **k: _fake_client(raw, record=sent))
    comp = courseapi._Completions(structured=(path == "structured"))
    msgs = [{"role": "user", "content": "hello"}]
    if path == "free_text":
        r = comp.create(messages=msgs, temperature=1.0)
    else:
        r = comp.parse(messages=msgs, response_format=_Schema, temperature=1.0)
    assert sent and sent[0]["max_output_tokens"] == config.MAX_OUTPUT_TOKENS
    assert r.provenance["transmitted"]["max_output_tokens"] == config.MAX_OUTPUT_TOKENS
    assert r.provenance["model_requested"] == config.MODEL
    assert r.provenance["model_served"] == "gpt-5.4-mini"
    assert r.provenance["input_sha256_16"]


def test_a_rate_limit_on_the_FREE_TEXT_path_is_waited_out(monkeypatch):
    """Free text had no backoff at all: a 429 raised a raw SDK error immediately."""
    import courseapi
    import unsw_ai
    calls = {"n": 0}
    raw = _Raw("completed")
    base = _fake_client(raw)

    def flaky(**kw):
        calls["n"] += 1
        if calls["n"] == 1:
            raise unsw_ai.QuotaExceededError("429 slow down", retry_after=0.0)
        return raw

    base.client.responses.create = flaky
    monkeypatch.setattr(unsw_ai, "get_client", lambda *a, **k: base)
    monkeypatch.setattr(courseapi.time, "sleep", lambda s: None)
    r = courseapi._Completions(structured=False).create(
        messages=[{"role": "user", "content": "hi"}], temperature=1.0)
    assert calls["n"] == 2, "the free-text path must wait out a rate limit, not raise"
    assert r.provenance["adapter_attempts"] == 2


def test_a_daily_quota_failure_is_NOT_waited_out(monkeypatch):
    """An exhausted daily budget will not clear in seconds; retrying just burns time."""
    import courseapi
    import unsw_ai
    base = _fake_client(_Raw("completed"))

    def dead(**kw):
        raise unsw_ai.DailyQuotaExceededError("daily budget gone")

    base.client.responses.create = dead
    monkeypatch.setattr(unsw_ai, "get_client", lambda *a, **k: base)
    monkeypatch.setattr(courseapi.time, "sleep",
                        lambda s: pytest.fail("must not back off on a daily quota"))
    with pytest.raises(unsw_ai.DailyQuotaExceededError):
        courseapi._Completions(structured=False).create(
            messages=[{"role": "user", "content": "hi"}], temperature=1.0)


# -------------------------------------------------------------------------------------------
# ONE TRANSPORT BUDGET, END TO END  (recheck-2 F2)
# -------------------------------------------------------------------------------------------

def _mock_stack(monkeypatch, responder):
    """The REAL adapter/instructor/SDK stack over a MockTransport, and the request log."""
    import httpx
    import openai
    import courseapi
    import unsw_ai

    sent = []

    def transport(request):
        sent.append(request)
        return responder(request)

    settings = unsw_ai.ProxySettings(proxy_url="https://mock.invalid",
                                     access_code="synthetic", student_id="9999999",
                                     fallback_models=())
    sdk = openai.OpenAI(api_key="synthetic", base_url="https://mock.invalid",
                        max_retries=0,
                        http_client=httpx.Client(transport=httpx.MockTransport(transport)))
    monkeypatch.setattr(unsw_ai, "build_openai_client", lambda *a, **k: sdk)
    monkeypatch.setattr(unsw_ai.time, "sleep", lambda *a, **k: None)
    client = unsw_ai.UNSWInstructor(settings=settings)
    monkeypatch.setattr(unsw_ai, "get_client", lambda *a, **k: client)
    monkeypatch.setattr(courseapi, "client_", lambda: courseapi.CourseClient())
    return sent


def _mock_response(status="completed", reason=None, content=None):
    import httpx
    import config as _c
    body = content if content is not None else [
        {"type": "output_text", "text": '{"value":', "annotations": []}]
    return httpx.Response(200, json={
        "id": "resp_synthetic", "object": "response", "created_at": 1,
        "status": status,
        "incomplete_details": {"reason": reason} if status == "incomplete" else None,
        "model": _c.MODEL,
        "output": [{"type": "message", "id": "msg_synthetic", "status": status,
                    "role": "assistant", "content": body}],
        "usage": {"input_tokens": 10, "output_tokens": 10, "total_tokens": 20}})


def test_a_persistent_rate_limit_costs_ONE_budget_not_one_per_layer(monkeypatch, tmp_path):
    """
    THE DEFECT THIS PINS. The adapter's "exactly one retry budget" was true of the adapter
    only: Words, Replay and Shock each wrapped config.MAX_RETRIES attempts around the
    wrapper's rate-limit budget, so ONE persistent 429 cost 5 x 4 = 20 HTTP requests for a
    single logical draw against a 60/minute budget shared by the whole class.
    """
    import httpx
    import text_features as tf
    import unsw_ai

    sent = _mock_stack(monkeypatch, lambda r: httpx.Response(
        429, json={"error": {"message": "Rate limit reached",
                             "type": "rate_limit_error", "code": "rate_limit"}}))
    monkeypatch.setattr(tf, "_schema_model", lambda: _Schema)
    monkeypatch.setattr(tf.time, "sleep", lambda *a, **k: None)
    envelope = tf.score_once("synthetic", config.CALL_INDEX_BASE)

    budget = unsw_ai.DEFAULT_RATE_LIMIT_RETRIES + 1
    assert len(sent) == budget, (
        f"one persistent 429 cost {len(sent)} HTTP requests; the wrapper's budget is "
        f"{budget} and no layer above it may open a second one")
    assert envelope["failure"]["category"] == "rate_limit"


def test_a_model_refusal_is_not_re_asked(monkeypatch):
    """A refusal is a COMPLETED answer. Repairing it buys a second identical refusal."""
    import text_features as tf

    sent = _mock_stack(monkeypatch, lambda r: _mock_response(
        status="completed",
        content=[{"type": "refusal", "refusal": "Synthetic refusal"}]))
    monkeypatch.setattr(tf, "_schema_model", lambda: _Schema)
    monkeypatch.setattr(tf.time, "sleep", lambda *a, **k: None)
    envelope = tf.score_once("synthetic", config.CALL_INDEX_BASE)

    assert len(sent) == 1, f"a refusal was requested {len(sent)} times"
    assert envelope["failure"]["category"] == "refusal", envelope["failure"]


@pytest.mark.parametrize("status,reason,category", [
    ("incomplete", "max_output_tokens", "truncated"),
    ("incomplete", "content_filter", "content_filter"),
    ("in_progress", None, "nonterminal"),
])
def test_each_unfinished_response_is_classified_as_itself(monkeypatch, status, reason,
                                                          category):
    """
    `truncated`, `content_filter`, `refusal` and `nonterminal` all used to arrive as the
    string "IncompleteResponseError" and be told apart by substring. They need four
    different answers: a raised ceiling retries only the first.
    """
    import text_features as tf

    sent = _mock_stack(monkeypatch, lambda r: _mock_response(status=status, reason=reason))
    monkeypatch.setattr(tf, "_schema_model", lambda: _Schema)
    monkeypatch.setattr(tf.time, "sleep", lambda *a, **k: None)
    envelope = tf.score_once("synthetic", config.CALL_INDEX_BASE)

    assert len(sent) == 1
    assert envelope["failure"]["category"] == category, envelope["failure"]


# -------------------------------------------------------------------------------------------
# FAILED ENVELOPES CARRY THEIR SETTINGS  (recheck-2 F3)
# -------------------------------------------------------------------------------------------

def _ok_draw(idx, **over):
    import text_features as tf
    rec = {"ok": True, "parsed": {"ok": True}, "call_index": idx,
           "config_hash": tf.call_config_hash(),
           "model": config.MODEL, "temperature": config.SAMPLING_TEMPERATURE,
           "prompt_hash": tf._prompt_hash(),
           "request": {"model": config.MODEL,
                       "temperature": config.SAMPLING_TEMPERATURE,
                       "max_output_tokens": config.MAX_OUTPUT_TOKENS},
           "usage": {"prompt_tokens": 10, "completion_tokens": 1},
           "timestamp": "2026-09-12T00:00:00+00:00",
           "envelope_version": 2}
    rec.update(over)
    return rec


def test_a_real_truncation_stays_settled_at_an_unchanged_ceiling(monkeypatch, tmp_path):
    """
    THE REGRESSION THIS PINS. The ceiling-compatibility rule needs the ceiling the draw
    RAN UNDER, and failure envelopes did not record one - so re-running five genuine
    truncations at an unchanged ceiling made five fresh calls every time.
    """
    import text_features as tf

    sent = _mock_stack(monkeypatch, lambda r: _mock_response(
        status="incomplete", reason="max_output_tokens"))
    monkeypatch.setattr(tf, "_schema_model", lambda: _Schema)
    monkeypatch.setattr(tf, "run_dir", lambda: tmp_path)
    monkeypatch.setattr(tf.time, "sleep", lambda *a, **k: None)
    failed = tf.score_once("synthetic", config.CALL_INDEX_BASE)

    assert failed["request"]["max_output_tokens"] == config.MAX_OUTPUT_TOKENS, (
        "a failed envelope must record the ceiling it ran under")
    indices = [config.CALL_INDEX_BASE + i for i in range(config.N_PARALLEL_CALLS)]
    (tmp_path / "2020-06-01.json").write_text(
        json.dumps([dict(failed, call_index=i) for i in indices]))

    calls = {"n": 0}

    def counted(text, idx):
        calls["n"] += 1
        return dict(failed, call_index=idx)

    monkeypatch.setattr(tf, "score_once", counted)
    tf.score_document("2020-06-01", "synthetic")
    assert calls["n"] == 0, (
        f"{calls['n']} settled truncations were re-asked at an unchanged ceiling")


def test_a_programming_fault_aborts_instead_of_settling_the_document(monkeypatch, tmp_path):
    """
    THE CACHE-POISONING THIS PINS. A TypeError from the adapter was written as five
    settled failures per document; repairing the BUG then produced zero fresh calls and
    zero valid draws, because the configuration had not changed and every index looked
    resolved. A fault in this repository is a property of the run, not of the document.
    """
    import text_features as tf

    def broken(**kw):
        raise TypeError("synthetic adapter configuration bug")

    monkeypatch.setattr(tf, "_schema_model", lambda: _Schema)
    monkeypatch.setattr(tf, "run_dir", lambda: tmp_path)
    monkeypatch.setattr(tf.time, "sleep", lambda *a, **k: None)
    monkeypatch.setattr(tf, "client_", lambda: types.SimpleNamespace(
        beta=types.SimpleNamespace(chat=types.SimpleNamespace(
            completions=types.SimpleNamespace(parse=broken)))))
    tf._RUN_ABORTED.clear()
    with pytest.raises(TypeError):
        tf.score_document("2020-01-01", "synthetic")

    cached = json.loads((tmp_path / "2020-01-01.json").read_text()) \
        if (tmp_path / "2020-01-01.json").exists() else []
    assert not [c for c in cached if tf._settled_failure(c)], (
        "a programming fault was written as a settled document failure")

    tf._RUN_ABORTED.clear()
    calls = {"n": 0}

    def repaired(text, idx):
        calls["n"] += 1
        return _ok_draw(idx)

    monkeypatch.setattr(tf, "score_once", repaired)
    rec = tf.score_document("2020-01-01", "synthetic")
    assert calls["n"] == config.N_PARALLEL_CALLS
    assert sum(1 for c in rec["calls"] if c["ok"]) == config.N_PARALLEL_CALLS, (
        "repairing the bug must recover the document")


# -------------------------------------------------------------------------------------------
# THE CACHE VALIDATES AN ENVELOPE AGAINST ITSELF  (recheck-2 F7)
# -------------------------------------------------------------------------------------------

@pytest.mark.parametrize("field,value", [
    ("model", "some-other-model"),
    ("temperature", 9),
    ("prompt_hash", "not-this-rubric"),
    ("model_served", "some-other-model"),
])
def test_a_cached_draw_that_contradicts_itself_is_refused(monkeypatch, tmp_path, field,
                                                          value):
    """
    A MATCHING config_hash IS NOT PROVENANCE. A synthetic cache naming a different model,
    a temperature of 9, an unrelated prompt and an `incomplete` response status loaded and
    scored, because nothing compared the envelope's own fields against the run.
    """
    import text_features as tf

    monkeypatch.setattr(tf, "_schema_model", lambda: _Schema)
    monkeypatch.setattr(tf, "run_dir", lambda: tmp_path)
    indices = [config.CALL_INDEX_BASE + i for i in range(config.N_PARALLEL_CALLS)]
    (tmp_path / "2020-05-01.json").write_text(
        json.dumps([_ok_draw(i, **{field: value}) for i in indices]))
    with pytest.raises(RuntimeError, match="offline"):
        tf.score_document("2020-05-01", "synthetic", offline=True)


def test_a_success_recorded_as_incomplete_is_refused(monkeypatch, tmp_path):
    import text_features as tf

    monkeypatch.setattr(tf, "_schema_model", lambda: _Schema)
    monkeypatch.setattr(tf, "run_dir", lambda: tmp_path)
    indices = [config.CALL_INDEX_BASE + i for i in range(config.N_PARALLEL_CALLS)]
    (tmp_path / "2020-05-02.json").write_text(json.dumps(
        [_ok_draw(i, provenance={"response_status": "incomplete"}) for i in indices]))
    with pytest.raises(RuntimeError, match="offline"):
        tf.score_document("2020-05-02", "synthetic", offline=True)


def test_a_legacy_envelope_without_transport_fields_still_loads(monkeypatch, tmp_path):
    """Absence in a legacy record is classified, never back-filled with an invented value."""
    import text_features as tf

    monkeypatch.setattr(tf, "_schema_model", lambda: _Schema)
    monkeypatch.setattr(tf, "run_dir", lambda: tmp_path)
    indices = [config.CALL_INDEX_BASE + i for i in range(config.N_PARALLEL_CALLS)]
    legacy = []
    for i in indices:
        rec = _ok_draw(i)
        for gone in ("envelope_version", "prompt_hash", "model", "temperature"):
            rec.pop(gone)
        legacy.append(rec)
    (tmp_path / "2020-05-03.json").write_text(json.dumps(legacy))
    out = tf.score_document("2020-05-03", "synthetic", offline=True)
    assert out["cached"] and len(out["calls"]) == config.N_PARALLEL_CALLS
    assert tf.envelope_generation(out["calls"][0]) == "legacy"


# -------------------------------------------------------------------------------------------
# THE PILOT AUDIT IS NAMED FOR THE AUDIT  (recheck-2 F8)
# -------------------------------------------------------------------------------------------

def test_a_gate_change_does_not_overwrite_the_previous_pilot_audit(monkeypatch, tmp_path):
    """
    Naming the file after `call_config_hash()` alone meant a run after a GATE change wrote
    to the same path - destroying the before-and-after pair the iteration marks are for.
    """
    import pandas as pd
    import text_features as tf

    docs = pd.DataFrame([{"meeting_date": "2020-01-01", "text_scored": "synthetic"}])
    monkeypatch.setattr(tf, "load_documents", lambda: docs)
    monkeypatch.setattr(tf, "pilot_meetings", lambda *a, **k: ["2020-01-01"])
    monkeypatch.setattr(tf, "_guard", lambda *a, **k: True)
    monkeypatch.setattr(tf, "_score_documents", lambda *a, **k: (
        [], {"api_calls": 0, "prompt_tokens": 0, "completion_tokens": 0}))
    monkeypatch.setattr(tf, "aggregate", lambda *a, **k: (
        pd.DataFrame(), {"checked": 1, "verbatim": 1}))
    monkeypatch.setattr(tf, "check_construct_quality", lambda *a, **k: pd.DataFrame(
        [{"construct": "synthetic", "passes_spread": True}]))
    monkeypatch.setattr(tf, "orientation_check", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(tf, "PILOT_DIR", tmp_path)
    monkeypatch.setattr(tf, "run_dir", lambda: tmp_path)

    tf.run_pilot(offline=True)
    first = sorted(tmp_path.glob("*.json"))
    assert len(first) == 1
    saved = json.loads(first[0].read_text())
    assert saved["gates"]["min_spread"] == config.MIN_CONSTRUCT_SPREAD, (
        "a saved audit must record the thresholds it was judged against")
    assert saved["audit_id"] and saved["stage_config_hash"] and saved["call_config_hash"]

    monkeypatch.setattr(config, "MIN_CONSTRUCT_SPREAD",
                        config.MIN_CONSTRUCT_SPREAD + 0.01)
    tf.run_pilot(offline=True)
    assert len(sorted(tmp_path.glob("*.json"))) == 2, (
        "the audit taken under the previous gate was overwritten")


# -------------------------------------------------------------------------------------------
# THE EXPOSURE LOG IS APPEND-ONLY  (recheck-2 F4)
# -------------------------------------------------------------------------------------------

@pytest.fixture()
def frozen(monkeypatch, tmp_path):
    monkeypatch.setattr(dr, "SELECTION_STAMP", tmp_path / "selection.json")
    monkeypatch.setattr(dr, "EXPOSURE_LOG", tmp_path / "exposures.jsonl")
    monkeypatch.setattr(dr, "dev_sample", lambda *a, **k: ["2020-01-01"])
    monkeypatch.setattr(dr, "holdout_sample", lambda *a, **k: ["2020-02-01"])
    return tmp_path


def test_refreezing_the_same_configuration_keeps_every_recorded_exposure(frozen):
    """
    THE DEFECT THIS PINS. `freeze_selection()` reset `exposure_number` to 0 and `exposures`
    to [] on EVERY freeze, so after two real exposures an ordinary same-configuration
    `--dev` freeze wiped the evidence - and the next changed-configuration freeze was then
    accepted without `--revalidate`, because the count it consulted was zero.
    """
    first = dr.freeze_selection()
    dr.record_holdout_exposure("synthetic first exposure")
    again = dr.freeze_selection()
    assert again["exposure_number"] == 1
    assert len(again["exposures"]) == 1
    assert again["freeze_id"] == first["freeze_id"], "a no-op freeze changed its identity"

    with pytest.raises(RuntimeError, match="already been run"):
        with _patched_draw_base():
            dr.freeze_selection()


class _patched_draw_base:
    def __enter__(self):
        self._old = config.CALL_INDEX_BASE
        config.CALL_INDEX_BASE = self._old + 100

    def __exit__(self, *exc):
        config.CALL_INDEX_BASE = self._old
        return False


def test_a_second_exposure_needs_declared_authority(frozen):
    dr.freeze_selection()
    event = dr.record_holdout_exposure("first")
    dr.close_holdout_exposure(event["event_id"], "benchmark completed")
    with pytest.raises(RuntimeError, match="authorises"):
        dr.record_holdout_exposure("a genuinely new question")

    with pytest.raises(RuntimeError, match="say why"):
        dr.freeze_selection(revalidate=True)
    dr.freeze_selection(note="the reviewer asked for a second look", revalidate=True)
    second = dr.record_holdout_exposure("declared second exposure")
    assert second["sequence"] == 2
    assert dr.evidence_status() == "retrospective"


def test_resuming_an_open_exposure_is_not_a_new_exposure(frozen):
    dr.freeze_selection()
    first = dr.record_holdout_exposure("first")
    resumed = dr.record_holdout_exposure("resumed after a crash")
    assert resumed["event_id"] == first["event_id"] and resumed["resumed"]
    assert len(dr.exposures_recorded()) == 1


def test_deleting_the_selection_file_does_not_restore_exposure_authority(frozen):
    """The log is the authority on what the holdout has seen, not the freeze record."""
    dr.freeze_selection()
    dr.record_holdout_exposure("first")
    (frozen / "selection.json").unlink()
    with pytest.raises(RuntimeError, match="already been run"):
        with _patched_draw_base():
            dr.freeze_selection()


def test_an_undeclared_empty_log_is_unexposed_and_a_declared_one_is_not(frozen):
    """An empty log cannot tell "untouched" from "exposed before the log existed"."""
    dr.freeze_selection()
    assert dr.evidence_status() == "unexposed"
    dr.freeze_selection(note="ran before the log existed", prior_exposure=True)
    assert dr.evidence_status() == "previously_exposed"
