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
import pathlib
import sys
import threading
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
    # A REAL RESPONSE ALWAYS CARRIES THESE, so a fake that omits them is not a fake of
    # this adapter - it is a fake of a defect. This one used to report `usage=None` and
    # no transport count, which is exactly the shape the envelope contract exists to
    # reject, so the fixture would have hidden a writer that stopped recording them.
    usage = {"input_tokens": 10, "output_tokens": 10, "total_tokens": 20,
             "prompt_tokens": 10, "completion_tokens": 10, "reasoning_tokens": 0,
             "cached_input_tokens": 0}
    return types.SimpleNamespace(
        choices=[types.SimpleNamespace(message=msg)],
        id="fake-request",
        usage=dict(usage),
        call_usage=dict(usage, attempts_counted=1, attempts_unknown=0,
                        tokens_known=True, is_complete=True),
        request={"model": config.MODEL, "temperature": config.SAMPLING_TEMPERATURE,
                 "max_output_tokens": config.MAX_OUTPUT_TOKENS},
        # `transmitted` is the COMPLETE request, exactly as `_provenance` copies it. A fake
        # that recorded only the ceiling there was a fake of a writer that dropped the
        # model and temperature from its own record of what it sent - and it passed only
        # because nothing read that copy.
        provenance={"transmitted": {"model": config.MODEL,
                                    "temperature": config.SAMPLING_TEMPERATURE,
                                    "max_output_tokens": config.MAX_OUTPUT_TOKENS},
                    "model_requested": config.MODEL, "model_served": config.MODEL,
                    "response_status": "completed", "transport_requests": 1,
                    "input_sha256_16": "0123456789abcdef",
                    "envelope_version": 2},
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
    # ONLY THE NETWORK IS SCRIPTED. The wrapper's own ProxyTransport and its token guard run
    # for real; replacing the whole transport hid every request the client refuses itself,
    # which is exactly where request counting went wrong.
    built = []

    def build(settings, **kwargs):
        proxy = unsw_ai.ProxyTransport(settings, inner=httpx.MockTransport(transport),
                                       token_limiter=kwargs.get("token_limiter"))
        built.append(openai.OpenAI(api_key="synthetic", base_url="https://mock.invalid",
                                   max_retries=0, http_client=httpx.Client(transport=proxy)))
        return built[-1]

    monkeypatch.setattr(unsw_ai, "build_openai_client", build)
    monkeypatch.setattr(unsw_ai.time, "sleep", lambda *a, **k: None)
    client = unsw_ai.UNSWInstructor(settings=settings)
    sdk = built[-1]
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
    """A CURRENT success envelope, complete to the contract.

    This helper used to omit the response-side provenance a real writer always records -
    exactly the shape the contract exists to reject - so it would have kept passing if
    the writer stopped recording them.
    """
    import courseapi
    import text_features as tf
    rec = {"ok": True, "parsed": {"ok": True}, "call_index": idx,
           "config_hash": tf.call_config_hash(),
           "model": config.MODEL, "temperature": config.SAMPLING_TEMPERATURE,
           "model_served": config.MODEL,
           "prompt_hash": tf._prompt_hash(),
           "request": {"model": config.MODEL,
                       "temperature": config.SAMPLING_TEMPERATURE,
                       "max_output_tokens": config.MAX_OUTPUT_TOKENS},
           "provenance": {"response_status": "completed",
                          "model_requested": config.MODEL,
                          "model_served": config.MODEL,
                          "input_sha256_16": "0123456789abcdef",
                          "transport_requests": 1},
           "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11,
                     "attempts_counted": 1, "attempts_unknown": 0,
                     "tokens_known": True, "is_complete": True},
           "transport_requests": 1,
           "timestamp": "2026-09-12T00:00:00+00:00",
           "envelope_version": courseapi.ENVELOPE_VERSION}
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


# -------------------------------------------------------------------------------------------
# EVERY FAILURE THE WRAPPER CAN RAISE IS CLASSIFIED  (recheck-3 N3)
# -------------------------------------------------------------------------------------------

def test_every_wrapper_error_has_an_explicit_category():
    """
    THE REGRESSION THIS PINS. The category map was written from the SDK's exception names,
    but the wrapper translates those into its own before the adapter sees them - so
    `APITimeoutError` became `ProxyTimeoutError` and `InternalServerError` became
    `UpstreamServiceError`, neither was in the map, both were classified `unknown`, and
    `unknown` is treated as a spent budget. Two transient failures that exist to be
    retried stopped the run after one attempt.

    A hand-written list will drift again unless something checks it, so this enumerates
    the wrapper's actual exception classes instead of restating the list.
    """
    import courseapi
    import unsw_ai

    unclassified = []
    for name in dir(unsw_ai):
        klass = getattr(unsw_ai, name)
        if not (isinstance(klass, type) and issubclass(klass, unsw_ai.UNSWAIError)):
            continue
        if klass is unsw_ai.UNSWAIError:
            continue
        try:
            instance = klass("synthetic")
        except Exception:                                # pragma: no cover
            continue
        if courseapi.failure_category(instance) == "unknown":
            unclassified.append(name)
    assert not unclassified, (
        f"these wrapper errors have no category, so they would be treated as unknown and "
        f"their retry budget as spent: {sorted(unclassified)}. Add each to "
        f"courseapi._CATEGORY_BY_NAME.")


def test_every_category_is_in_the_declared_vocabulary():
    import courseapi
    stray = sorted(set(courseapi._CATEGORY_BY_NAME.values())
                   - set(courseapi.FAILURE_CATEGORIES))
    assert not stray, f"categories not declared in FAILURE_CATEGORIES: {stray}"
    for group in (courseapi.SETTLED_CATEGORIES, courseapi.RUN_LEVEL_CATEGORIES,
                  courseapi.STAGE_RETRYABLE_CATEGORIES,
                  courseapi.PRETRANSPORT_CATEGORIES):
        stray = sorted(set(group) - set(courseapi.FAILURE_CATEGORIES))
        assert not stray, f"undeclared categories in a policy group: {stray}"
    overlap = sorted(set(courseapi.SETTLED_CATEGORIES)
                     & set(courseapi.RUN_LEVEL_CATEGORIES))
    assert not overlap, (
        f"a category cannot be both a settled document outcome and a run-level abort: "
        f"{overlap}")


# -------------------------------------------------------------------------------------------
# TRANSIENT FAILURES RECOVER; PERSISTENT ONES STOP  (recheck-3 N3, plan T4)
# -------------------------------------------------------------------------------------------

def _sequence_transport(monkeypatch, responses):
    """Serve `responses` in order; each entry is a callable taking the httpx request."""
    import httpx
    import openai
    import courseapi
    import unsw_ai

    sent = []

    def transport(request):
        sent.append(request)
        make = responses[min(len(sent) - 1, len(responses) - 1)]
        return make(request)

    settings = unsw_ai.ProxySettings(proxy_url="https://mock.invalid",
                                     access_code="synthetic", student_id="9999999",
                                     fallback_models=())
    # ONLY THE NETWORK IS SCRIPTED. The wrapper's own ProxyTransport and its token guard run
    # for real; replacing the whole transport hid every request the client refuses itself,
    # which is exactly where request counting went wrong.
    built = []

    def build(settings, **kwargs):
        proxy = unsw_ai.ProxyTransport(settings, inner=httpx.MockTransport(transport),
                                       token_limiter=kwargs.get("token_limiter"))
        built.append(openai.OpenAI(api_key="synthetic", base_url="https://mock.invalid",
                                   max_retries=0, http_client=httpx.Client(transport=proxy)))
        return built[-1]

    monkeypatch.setattr(unsw_ai, "build_openai_client", build)
    monkeypatch.setattr(unsw_ai.time, "sleep", lambda *a, **k: None)
    client = unsw_ai.UNSWInstructor(settings=settings)
    sdk = built[-1]
    monkeypatch.setattr(unsw_ai, "get_client", lambda *a, **k: client)
    monkeypatch.setattr(courseapi, "client_", lambda: courseapi.CourseClient())
    return sent, sdk


def _tool_body(arguments, tokens, status="completed"):
    """One Responses-API tool call for `_Schema`, as instructor expects to parse it."""
    import config as _c
    return {
        "id": "resp_ok", "object": "response", "created_at": 1, "status": status,
        "incomplete_details": None, "model": _c.MODEL,
        "output": [{"type": "function_call", "id": "fc", "call_id": "c",
                    "name": _Schema.__name__, "arguments": arguments,
                    "status": "completed"}],
        "usage": {"input_tokens": 0, "output_tokens": tokens, "total_tokens": tokens}}


def _tool_response(tokens=20):
    import httpx
    return lambda request: httpx.Response(
        200, json=_tool_body(json.dumps({"ok": True}), tokens))


def _http_error(code):
    import httpx
    return lambda request: httpx.Response(
        code, json={"error": {"message": "synthetic", "type": "server_error"}})


def _read_timeout():
    import httpx
    def raise_timeout(request):
        raise httpx.ReadTimeout("synthetic timeout", request=request)
    return raise_timeout


@pytest.mark.parametrize("first,label", [
    (_http_error(500), "HTTP 500"),
    (_http_error(503), "HTTP 503"),
    (_read_timeout(), "read timeout"),
])
def test_a_transient_failure_then_success_costs_exactly_two_requests(monkeypatch, first,
                                                                     label):
    """
    Through the WHOLE stack - SDK, instructor, the wrapper's translation, this adapter and
    the stage's own retry. Raising a hand-made SDK exception at the adapter would skip the
    translation step that caused the defect, and would have passed while the real path
    stopped after one attempt.
    """
    import text_features as tf

    sent, sdk = _sequence_transport(monkeypatch, [first, _tool_response()])
    monkeypatch.setattr(tf, "_schema_model", lambda: _Schema)
    monkeypatch.setattr(tf.time, "sleep", lambda *a, **k: None)
    try:
        envelope = tf.score_once("synthetic", config.CALL_INDEX_BASE)
    finally:
        sdk.close()

    assert envelope["ok"] is True, (
        f"a {label} followed by a good response did not recover: {envelope.get('error')}")
    assert len(sent) == 2, f"{label} then success cost {len(sent)} requests, not 2"


@pytest.mark.parametrize("make,label", [
    (_http_error(500), "HTTP 500"),
    (_read_timeout(), "read timeout"),
])
def test_a_persistent_transient_failure_stops_at_the_declared_bound(monkeypatch, make,
                                                                    label):
    import text_features as tf

    sent, sdk = _sequence_transport(monkeypatch, [make])
    monkeypatch.setattr(tf, "_schema_model", lambda: _Schema)
    monkeypatch.setattr(tf.time, "sleep", lambda *a, **k: None)
    try:
        envelope = tf.score_once("synthetic", config.CALL_INDEX_BASE)
    finally:
        sdk.close()

    assert envelope["failure"]["category"] == "transport", (
        f"{label} was classified {envelope['failure']['category']!r}")
    assert len(sent) == config.MAX_RETRIES, (
        f"a persistent {label} cost {len(sent)} requests; the stage bound is "
        f"{config.MAX_RETRIES}")
    assert envelope["transport_requests"] == len(sent)


# -------------------------------------------------------------------------------------------
# EVERY KNOWN ATTEMPT IS ON THE BILL  (recheck-3 N4, plan T5)
# -------------------------------------------------------------------------------------------

def _bad_then_good(bad_tokens=20, good_tokens=30):
    """A completed-but-invalid object, then a valid one: instructor's repair case."""
    import httpx
    state = {"n": 0}

    def respond(request):
        state["n"] += 1
        first = state["n"] == 1
        arguments = ('{"ok":"not-a-boolean"}' if first else '{"ok":true}')
        return httpx.Response(
            200, json=_tool_body(arguments, bad_tokens if first else good_tokens))
    return respond


def test_a_repaired_success_bills_both_attempts(monkeypatch):
    """
    THE UNDERCOUNT THIS PINS. The envelope kept the LAST response's usage, so a draw that
    cost a 20-token invalid attempt and a 30-token good one reported 30. Understating a
    shared allowance is the wrong direction to be wrong in.
    """
    import text_features as tf

    sent, sdk = _sequence_transport(monkeypatch, [_bad_then_good()])
    monkeypatch.setattr(tf, "_schema_model", lambda: _Schema)
    monkeypatch.setattr(tf.time, "sleep", lambda *a, **k: None)
    try:
        envelope = tf.score_once("synthetic", config.CALL_INDEX_BASE)
    finally:
        sdk.close()

    assert envelope["ok"] is True, envelope.get("error")
    assert len(sent) == 2
    assert envelope["usage"]["total_tokens"] == 50, (
        f"two billed attempts of 20 and 30 tokens were recorded as "
        f"{envelope['usage']['total_tokens']}")
    assert envelope["usage"]["attempts_counted"] == 2
    assert envelope["usage_final_response"]["total_tokens"] == 30, (
        "the final response's own figure must stay available and separate")
    assert envelope["transport_requests"] == 2


def test_a_failed_free_text_call_keeps_the_tokens_it_spent(monkeypatch):
    """
    Free-text validation used to run AFTER the guard returned, so the one failure that
    knew exactly what it had cost threw the figure away: no transport count, no usage.
    """
    import courseapi
    import httpx

    def truncated(request):
        return httpx.Response(200, json={
            "id": "resp", "object": "response", "created_at": 1, "status": "incomplete",
            "incomplete_details": {"reason": "max_output_tokens"}, "model": config.MODEL,
            "output": [{"type": "message", "id": "m", "role": "assistant",
                        "status": "incomplete",
                        "content": [{"type": "output_text", "text": "partial",
                                     "annotations": []}]}],
            "usage": {"input_tokens": 10, "output_tokens": 10, "total_tokens": 20}})

    sent, sdk = _sequence_transport(monkeypatch, [truncated])
    try:
        with pytest.raises(courseapi.IncompleteResponseError) as caught:
            courseapi.CourseClient().chat.completions.create(
                messages=[{"role": "user", "content": "synthetic"}])
    finally:
        sdk.close()

    assert courseapi.transport_attempts(caught.value) == 1
    usage = courseapi.failed_attempt_usage(caught.value)
    assert usage and usage["total_tokens"] == 20, (
        f"a truncated free-text answer was billed 20 tokens and recorded {usage}")
    assert usage["tokens_known"] is True


def test_an_attempt_with_no_reported_usage_is_labelled_not_zeroed(monkeypatch):
    """
    A timeout costs an unknown amount, not nothing. Recording zero would let a run report
    a total it cannot support; the count of unreported attempts is carried instead.
    """
    import text_features as tf

    sent, sdk = _sequence_transport(
        monkeypatch, [_read_timeout(), _tool_response(tokens=30)])
    monkeypatch.setattr(tf, "_schema_model", lambda: _Schema)
    monkeypatch.setattr(tf.time, "sleep", lambda *a, **k: None)
    try:
        envelope = tf.score_once("synthetic", config.CALL_INDEX_BASE)
    finally:
        sdk.close()

    assert envelope["ok"] is True
    assert envelope["transport_requests"] == 2
    usage = envelope["usage"]
    assert usage["total_tokens"] == 30, "the known part of the bill must be kept"
    assert usage["attempts_unknown"] == 1, "the unreported attempt must be counted"
    assert usage["is_complete"] is False, "and the total must not claim to be complete"


def test_a_persistent_rate_limit_reports_unknown_cost_not_zero_tokens(monkeypatch):
    import text_features as tf

    sent, sdk = _sequence_transport(monkeypatch, [_http_error(429)])
    monkeypatch.setattr(tf, "_schema_model", lambda: _Schema)
    monkeypatch.setattr(tf.time, "sleep", lambda *a, **k: None)
    try:
        envelope = tf.score_once("synthetic", config.CALL_INDEX_BASE)
    finally:
        sdk.close()

    usage = envelope["usage"]
    assert usage is not None and usage["tokens_known"] is False, (
        "five billed-but-unreported attempts must not read as a known total")
    assert "total_tokens" not in usage, (
        "an unknown cost must not be recorded as a number that can be summed")
    assert usage["attempts_unknown"] == unsw_ai_budget()


def unsw_ai_budget():
    import unsw_ai
    return unsw_ai.DEFAULT_RATE_LIMIT_RETRIES + 1


# -------------------------------------------------------------------------------------------
# A RUN-CONFIGURATION FAULT ABORTS AND RECOVERS  (recheck-3 N2, plan T3)
# -------------------------------------------------------------------------------------------

def test_a_forbidden_model_fallback_aborts_before_any_request_and_recovers(monkeypatch,
                                                                          tmp_path):
    """
    THE CACHE POISONING THIS PINS. The adapter refuses a client with model fallback
    enabled - correctly - but the refusal was raised as `ParameterNotSupportedError`,
    which classifies as `request_rejected`: a SETTLED DOCUMENT OUTCOME. All five draws
    were written as settled failures, and correcting the client configuration then made
    zero fresh calls and kept zero valid draws, exactly as a mistyped access code once
    did. How the run is configured is not what the request asked for.
    """
    import courseapi
    import text_features as tf

    requests = {"n": 0}

    def never_called(**kwargs):
        requests["n"] += 1
        raise AssertionError("a forbidden fallback must be refused before any request")

    misconfigured = types.SimpleNamespace(
        settings=types.SimpleNamespace(fallback_models=("some-other-model",)),
        beta=types.SimpleNamespace(chat=types.SimpleNamespace(
            completions=types.SimpleNamespace(parse=never_called))),
        chat=types.SimpleNamespace(completions=types.SimpleNamespace(
            create_with_completion=never_called)),
        client=types.SimpleNamespace(responses=types.SimpleNamespace(
            create=never_called)))

    monkeypatch.setattr(tf, "_schema_model", lambda: _Schema)
    monkeypatch.setattr(tf, "run_dir", lambda: tmp_path)
    monkeypatch.setattr(tf.time, "sleep", lambda *a, **k: None)

    indices = [config.CALL_INDEX_BASE + i for i in range(config.N_PARALLEL_CALLS)]
    survivor = _ok_draw(indices[0])
    (tmp_path / "2020-05-01.json").write_text(json.dumps([survivor]))

    tf._RUN_ABORTED.clear()
    monkeypatch.setattr(courseapi.unsw_ai, "get_client", lambda *a, **k: misconfigured)
    monkeypatch.setattr(tf, "client_", lambda: courseapi.CourseClient())
    with pytest.raises(courseapi.RunConfigurationError):
        tf.score_document("2020-05-01", "synthetic")

    assert requests["n"] == 0, "the refusal must happen before any HTTP request"
    cached = json.loads((tmp_path / "2020-05-01.json").read_text())
    assert not [c for c in cached if tf._settled_failure(c)], (
        "a run-configuration fault was written as a settled document outcome")
    assert survivor in cached, "the draw that already succeeded must survive untouched"

    # The configuration is repaired. No cache is deleted by hand.
    tf._RUN_ABORTED.clear()
    calls = {"n": 0}

    def repaired(text, idx):
        calls["n"] += 1
        return _ok_draw(idx)

    monkeypatch.setattr(tf, "score_once", repaired)
    rec = tf.score_document("2020-05-01", "synthetic")
    assert calls["n"] == config.N_PARALLEL_CALLS - 1, (
        "exactly the missing draws must be requested - not the one that already worked")
    assert sum(1 for c in rec["calls"] if c.get("ok")) == config.N_PARALLEL_CALLS


# -------------------------------------------------------------------------------------------
# THE ENVELOPE CONTRACT, WRITER TO LOADER, IN ALL THREE STAGES  (recheck-3 N5, plan T6/T9)
# -------------------------------------------------------------------------------------------

def _contract_response():
    import courseapi
    request = courseapi.effective_request(
        model=config.MODEL, temperature=config.SAMPLING_TEMPERATURE,
        max_tokens=config.MAX_OUTPUT_TOKENS)
    usage = {"input_tokens": 10, "output_tokens": 10, "total_tokens": 20}
    return types.SimpleNamespace(
        choices=[types.SimpleNamespace(
            message=types.SimpleNamespace(parsed=_Schema(), content="synthetic text"))],
        request=request, model=config.MODEL, id="resp_synthetic",
        usage=dict(usage),
        call_usage=dict(usage, attempts_counted=1, attempts_unknown=0,
                        tokens_known=True, is_complete=True),
        # `transmitted` is the complete request, exactly as `_provenance` writes it. Without
        # it, the loader's comparison of that copy could never be exercised through here.
        provenance={"envelope_version": 2, "response_status": "completed",
                    "transmitted": dict(request),
                    "model_requested": config.MODEL, "model_served": config.MODEL,
                    "input_sha256_16": "0123456789abcdef",
                    "transport_requests": 1})


@pytest.mark.parametrize("stage", ["replay", "shock"])
def test_replay_and_shock_write_and_enforce_the_shared_envelope_contract(monkeypatch,
                                                                        tmp_path, stage):
    """
    THE ASYMMETRY THIS PINS. Words carried a version marker and checked contradictions;
    Replay and Shock wrote no version at all - so evidence written a minute ago
    classified itself as "legacy" - and their loaders compared only the request-side
    fields, so a cache edited to name a different served model with an `incomplete`
    response status was reused by both.
    """
    import decision_replay as dr
    import scenarios as sc

    mod = dr if stage == "replay" else sc
    response = _contract_response()
    client = types.SimpleNamespace(beta=types.SimpleNamespace(chat=types.SimpleNamespace(
        completions=types.SimpleNamespace(parse=lambda **kw: response))))
    call = ((lambda: dr._cached_call("probe", "system", "user", schema=_Schema))
            if stage == "replay" else
            (lambda: sc.llm_parsed("system", "user", _Schema)))

    monkeypatch.setattr(mod, "RAW_DIR", tmp_path)
    monkeypatch.setattr(mod, "client_", lambda: client)
    call()
    paths = list(tmp_path.glob("*.json"))
    assert len(paths) == 1, f"{stage} wrote {len(paths)} cache files"
    path, saved = paths[0], json.loads(paths[0].read_text())
    pristine = path.read_text()

    import courseapi
    assert saved.get("envelope_version") == courseapi.ENVELOPE_VERSION, (
        f"{stage} writes no envelope version, so its own fresh evidence reads as legacy")
    assert saved.get("usage") is not None and saved.get("transport_requests") is not None

    def reuses_cache():
        """True when the loader served the cache without calling."""
        monkeypatch.setattr(mod, "client_", _explode)
        try:
            value = call()
        except Exception:                                # noqa: BLE001
            return False
        # Replay returns a failed record rather than raising when it refuses a cache.
        return not (isinstance(value, dict) and value.get("ok") is False)

    assert reuses_cache(), "an unchanged valid record must replay without a request"

    for field, value in (("model_served", "other-model"), ("model", "other-model"),
                         ("temperature", 9), ("prompt_sha", "not-this-prompt"),
                         ("envelope_version", 99), ("envelope_version", True)):
        path.write_text(json.dumps(dict(saved, **{field: value})))
        assert not reuses_cache(), (
            f"{stage} reused a cached record whose {field} was changed to {value!r}")
        path.write_text(pristine)

    path.write_text(json.dumps(dict(
        saved, provenance=dict(saved["provenance"], response_status="incomplete"))))
    assert not reuses_cache(), (
        f"{stage} reused a record marked ok with a non-terminal response status")
    path.write_text(pristine)
    assert reuses_cache(), "the restored control must replay again"


def _explode(*a, **k):
    raise RuntimeError("this call must be served from cache")


# -------------------------------------------------------------------------------------------
# EVERY RECORDED COPY OF THE REQUEST, AND COUNTS AS COUNTS  (recheck-5 S2)
# -------------------------------------------------------------------------------------------
# Each case writes a VALID record through the real stage writer, over the real adapter and a
# scripted transport; proves the loader serves it from cache; changes ONE thing; and proves
# the loader then refuses it. Refusing means a fresh request was attempted AND no cached
# payload came back - for Replay a returned failed record is a refusal, not an acceptance,
# so the absence of an exception is never taken as a pass.

def _real_writer_cache(stage, tmp_path, monkeypatch):
    import warnings
    import httpx
    import openai
    from pydantic import BaseModel
    import courseapi
    import unsw_ai
    import decision_replay as dr
    import scenarios as sc

    class CacheProbe(BaseModel):
        value: int

    def respond(request):
        return httpx.Response(200, json={
            "id": "resp_synthetic", "object": "response", "created_at": 1,
            "model": config.MODEL, "status": "completed", "incomplete_details": None,
            "output": [{"type": "function_call", "id": "f", "call_id": "c",
                        "name": "CacheProbe", "arguments": '{"value":1}',
                        "status": "completed"}],
            "usage": {"input_tokens": 10, "output_tokens": 10, "total_tokens": 20}})

    mod = dr if stage == "replay" else sc
    settings = unsw_ai.ProxySettings(proxy_url="https://synthetic.invalid",
                                     access_code="synthetic", student_id="9999999",
                                     fallback_models=())

    def build(settings, **kwargs):
        # Only the network is scripted: the real ProxyTransport and token guard run.
        proxy = unsw_ai.ProxyTransport(settings, inner=httpx.MockTransport(respond),
                                       token_limiter=kwargs.get("token_limiter"))
        return openai.OpenAI(api_key="synthetic", base_url="https://synthetic.invalid",
                             max_retries=0, http_client=httpx.Client(transport=proxy))

    monkeypatch.setattr(unsw_ai, "build_openai_client", build)
    monkeypatch.setattr(unsw_ai.time, "sleep", lambda *a, **k: None)
    monkeypatch.setattr(config, "ledger_add", lambda *a, **k: None)
    monkeypatch.setattr(mod, "RAW_DIR", tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        client = unsw_ai.UNSWInstructor(settings=settings)
    monkeypatch.setattr(unsw_ai, "get_client", lambda *a, **k: client)
    monkeypatch.setattr(mod, "client_", lambda *a, **k: courseapi.CourseClient())
    if stage == "replay":
        def call():
            return dr._cached_call("probe", "system", "user", schema=CacheProbe)
    else:
        def call():
            return sc.llm_parsed("system", "user", CacheProbe)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        call()
    paths = sorted(tmp_path.glob("*.json"))
    assert len(paths) == 1, f"{stage} wrote {len(paths)} cache files"
    return paths[0], json.loads(paths[0].read_text(encoding="utf-8")), call, mod


def _served_from_cache(monkeypatch, mod, call):
    """True only when NOTHING was requested AND a successful payload came back."""
    attempts = {"n": 0}

    def refuse(*a, **k):
        attempts["n"] += 1
        raise RuntimeError("the cache was refused, so a fresh request was attempted")

    monkeypatch.setattr(mod, "client_", refuse)
    try:
        value = call()
    except Exception:                                    # noqa: BLE001
        value = None
    succeeded = value is not None and not (isinstance(value, dict)
                                           and value.get("ok") is False)
    return attempts["n"] == 0 and succeeded


def _without(block, key):
    block.pop(key, None)


_S2_MUTATIONS = {
    "request omits the temperature":
        lambda r: _without(r["request"], "temperature"),
    "transmitted omits the temperature":
        lambda r: _without(r["provenance"]["transmitted"], "temperature"),
    "transmitted temperature contradicts the run":
        lambda r: r["provenance"]["transmitted"].update(temperature=99),
    "transmitted model contradicts the run":
        lambda r: r["provenance"]["transmitted"].update(model="another-deployment"),
    "transmitted ceiling disagrees with the request":
        lambda r: r["provenance"]["transmitted"].update(
            max_output_tokens=r["request"]["max_output_tokens"] + 1),
    "negative transport count":
        lambda r: r.update(transport_requests=-1),
    "boolean transport count":
        lambda r: r.update(transport_requests=True),
    "a success that took no request":
        lambda r: r.update(transport_requests=0),
    "negative adapter attempts":
        lambda r: r["provenance"].update(adapter_attempts=-1),
    "negative token count":
        lambda r: r["usage"].update(total_tokens=-40),
    "token count that is not a number":
        lambda r: r["usage"].update(input_tokens="10"),
    "complete bill with attempts of unknown cost":
        lambda r: r["usage"].update(attempts_unknown=1),
    "unknown cost that still states tokens":
        lambda r: r["usage"].update(tokens_known=False),
    "explicit null version":
        lambda r: r.update(envelope_version=None),
    "null version with the current fields stripped":
        lambda r: r.update(envelope_version=None, provenance=None, model_served=None,
                           transport_requests=None, usage=None),
}


@pytest.mark.parametrize("stage", ["replay", "shock"])
@pytest.mark.parametrize("mutation", list(_S2_MUTATIONS))
def test_a_cached_record_with_one_invalid_field_is_not_served(stage, mutation, tmp_path,
                                                             monkeypatch):
    path, saved, call, mod = _real_writer_cache(stage, tmp_path, monkeypatch)
    assert _served_from_cache(monkeypatch, mod, call), (
        f"{stage}: the unmodified record the real writer produced must be served from "
        f"cache - without that control, a refusal below proves nothing")
    changed = json.loads(json.dumps(saved))
    _S2_MUTATIONS[mutation](changed)
    # Compared as WRITTEN, not as Python values: `True == 1`, so a boolean count compares
    # equal to the integer it replaces while the file on disk says `true`.
    assert json.dumps(changed, sort_keys=True) != json.dumps(saved, sort_keys=True), (
        "the mutation must actually change the record")
    path.write_text(json.dumps(changed), encoding="utf-8")
    assert not _served_from_cache(monkeypatch, mod, call), (
        f"{stage} served a cached payload from a record where the {mutation}")


@pytest.mark.parametrize("stage", ["replay", "shock"])
def test_genuine_legacy_evidence_is_still_served(stage, tmp_path, monkeypatch):
    """
    The control these rules must not break. Every committed Replay and Shock envelope in
    the worked exemplar has exactly this shape: no version, no provenance, no transport
    count, and a usage block of token figures only. Absence there is history, not a defect.
    """
    path, saved, call, mod = _real_writer_cache(stage, tmp_path, monkeypatch)
    legacy = {k: v for k, v in saved.items()
              if k not in ("envelope_version", "provenance", "model_served",
                           "transport_requests", "usage_final_response")}
    legacy["usage"] = {k: v for k, v in saved["usage"].items()
                       if k not in ("attempts_counted", "attempts_unknown",
                                    "tokens_known", "is_complete")}
    path.write_text(json.dumps(legacy), encoding="utf-8")
    assert _served_from_cache(monkeypatch, mod, call), (
        f"{stage} refused a genuine legacy record - the new checks belong to the current "
        f"contract and must never be applied to history")


@pytest.mark.parametrize("stage", ["replay", "shock"])
@pytest.mark.parametrize("fault", ["missing access code", "forbidden model fallback"])
def test_a_failure_that_sent_nothing_records_no_request(stage, fault, tmp_path,
                                                        monkeypatch):
    """
    Found while checking the AI-use template against what the writers emit: a missing
    access code and a forbidden model fallback were recorded as `transport_requests: 1`,
    because the stages counted a failure with no transport stamp as one request. Nothing
    was sent. The two faults are raised in different places - `get_client` before the
    guard, `_no_substitution` inside it - so both paths are exercised.
    """
    from pydantic import BaseModel
    import courseapi
    import unsw_ai
    import decision_replay as dr
    import scenarios as sc

    class CacheProbe(BaseModel):
        value: int

    if fault == "missing access code":
        def get_client(*a, **k):
            raise unsw_ai.MissingCredentialsError("synthetic: no access code")
    else:
        misconfigured = types.SimpleNamespace(
            settings=types.SimpleNamespace(fallback_models=("another-deployment",)))

        def get_client(*a, **k):
            return misconfigured

    mod = dr if stage == "replay" else sc
    monkeypatch.setattr(unsw_ai, "get_client", get_client)
    monkeypatch.setattr(unsw_ai.time, "sleep", lambda *a, **k: None)
    monkeypatch.setattr(config, "ledger_add", lambda *a, **k: None)
    monkeypatch.setattr(mod, "RAW_DIR", tmp_path)
    monkeypatch.setattr(mod, "client_", lambda *a, **k: courseapi.CourseClient())
    try:
        if stage == "replay":
            dr._cached_call("probe", "system", "user", schema=CacheProbe)
        else:
            sc.llm_parsed("system", "user", CacheProbe)
    except Exception:                                    # noqa: BLE001
        pass                       # both stages raise AFTER writing the envelope
    written = sorted(tmp_path.glob("*.json"))
    assert len(written) == 1, f"{stage} wrote {len(written)} envelopes for one call"
    rec = json.loads(written[0].read_text(encoding="utf-8"))
    assert rec["transport_requests"] is None, (
        f"{stage} recorded {rec['transport_requests']} request(s) for a call that sent none")
    assert rec["failure"]["transport_attempts"] == 0, (
        f"{stage}: failure.transport_attempts is {rec['failure']['transport_attempts']!r}; "
        f"a failure raised before any request left is zero requests, not an unknown")
    assert rec["usage"] is None, f"{stage} recorded a bill for a call that sent nothing"


@pytest.mark.parametrize("stage", ["replay", "shock"])
def test_a_run_without_a_sampling_temperature_is_validated_as_one(stage, tmp_path,
                                                                  monkeypatch):
    """Temperature is required WHILE the run samples - and refused when it does not."""
    monkeypatch.setattr(config, "SAMPLING_TEMPERATURE", None)
    path, saved, call, mod = _real_writer_cache(stage, tmp_path, monkeypatch)
    assert "temperature" not in saved["request"], saved["request"]
    assert _served_from_cache(monkeypatch, mod, call), (
        f"{stage}: a record written with no sampling temperature, under a run that sets "
        f"none, must be served")
    changed = json.loads(json.dumps(saved))
    changed["request"]["temperature"] = 1.0
    path.write_text(json.dumps(changed), encoding="utf-8")
    assert not _served_from_cache(monkeypatch, mod, call), (
        f"{stage} served a record claiming a temperature this run never sends")


@pytest.mark.parametrize("version,accepted", [
    (None, True),        # genuine legacy: absence is classified, never back-filled
    (2, True),           # the current contract
    (99, False),         # a future contract this code cannot read
    (True, False),       # a boolean is not a version
    ("2", False),        # nor is a string
])
def test_words_version_boundaries(monkeypatch, tmp_path, version, accepted):
    import text_features as tf

    monkeypatch.setattr(tf, "_schema_model", lambda: _Schema)
    monkeypatch.setattr(tf, "run_dir", lambda: tmp_path)
    indices = [config.CALL_INDEX_BASE + i for i in range(config.N_PARALLEL_CALLS)]
    draws = []
    for i in indices:
        rec = _ok_draw(i)
        if version is None:
            rec.pop("envelope_version")
        else:
            rec["envelope_version"] = version
        draws.append(rec)
    (tmp_path / "2020-08-01.json").write_text(json.dumps(draws))

    if accepted:
        out = tf.score_document("2020-08-01", "synthetic", offline=True)
        assert out["cached"] and len(out["calls"]) == config.N_PARALLEL_CALLS
    else:
        with pytest.raises(RuntimeError, match="offline"):
            tf.score_document("2020-08-01", "synthetic", offline=True)


def test_a_current_success_without_response_provenance_is_refused(monkeypatch, tmp_path):
    """A version-2 success with no provenance, served model, usage or transport count
    claims a contract it does not keep. Absence is a defect in a CURRENT record."""
    import text_features as tf

    monkeypatch.setattr(tf, "_schema_model", lambda: _Schema)
    monkeypatch.setattr(tf, "run_dir", lambda: tmp_path)
    indices = [config.CALL_INDEX_BASE + i for i in range(config.N_PARALLEL_CALLS)]
    for dropped in ("provenance", "model_served", "usage", "transport_requests"):
        draws = []
        for i in indices:
            rec = _ok_draw(i)
            rec.pop(dropped, None)
            draws.append(rec)
        (tmp_path / "2020-09-01.json").write_text(json.dumps(draws))
        with pytest.raises(RuntimeError, match="offline"):
            tf.score_document("2020-09-01", "synthetic", offline=True)


# -------------------------------------------------------------------------------------------
# THE EXPOSURE LOG IS THE AUTHORITY, AND THE VALIDATOR READS IT  (recheck-3 N1, plan T1/T2)
# -------------------------------------------------------------------------------------------

def _submission_module():
    import importlib.util
    here = pathlib.Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location(
        "submission_under_test", here / "test_submission.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def two_exposures(monkeypatch, tmp_path):
    """Exposure A, a DECLARED revalidation, then exposure B - the legitimate sequence.

    This is the workflow the design exists to support, and the first version of the
    submission check rejected it: it required every event in the history to carry the
    CURRENT selection hash, which is false of A by construction.
    """
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    monkeypatch.setattr(dr, "SELECTION_STAMP", outputs / "replay_selection.json")
    monkeypatch.setattr(dr, "EXPOSURE_LOG", outputs / "replay_exposures.jsonl")
    monkeypatch.setattr(dr, "dev_sample", lambda *a, **k: ["2020-01-01"])
    monkeypatch.setattr(dr, "holdout_sample", lambda *a, **k: ["2020-02-01"])

    dr.freeze_selection()
    a = dr.record_holdout_exposure("first exposure")
    dr.close_holdout_exposure(a["event_id"], "benchmark completed")

    base = config.CALL_INDEX_BASE
    monkeypatch.setattr(config, "CALL_INDEX_BASE", base + 100)
    dr.freeze_selection(revalidate=True, note="declared second experiment")
    b = dr.record_holdout_exposure("second exposure")
    dr.close_holdout_exposure(b["event_id"], "benchmark completed")

    sub = _submission_module()
    monkeypatch.setattr(sub, "ROOT", tmp_path)

    def artefact(event_id=None, events=None, flags=None, status=None):
        # Exactly what the real writer emits, flags included: a fixture that omits a
        # field the producer always writes is a fake of a defect, not of the producer.
        # `flags` and `status` exist so a test can mutate the SUMMARY while leaving the
        # files it summarises untouched - the shape of the defect S1 reported.
        real_flags = dr.evidence_flags()
        return {"selection": dr._read_selection(),
                "exposure": {
                    "events": dr.exposure_events() if events is None else events,
                    "evidence_status": status or real_flags["status"],
                    "evidence_flags": real_flags if flags is None else flags,
                    "holdout_event_id": event_id or b["event_id"]}}

    def validate(**kw):
        monkeypatch.setattr(sub, "_artefact", lambda name: artefact(**kw))
        sub.test_the_holdout_result_belongs_to_a_frozen_selection_and_a_recorded_exposure()

    return types.SimpleNamespace(a=a, b=b, log=dr.EXPOSURE_LOG, validate=validate,
                                 artefact=artefact, sub=sub, tmp=tmp_path)


def test_a_declared_revalidation_passes_the_submission_check(two_exposures):
    """The control. Every rejection test below starts from this passing state."""
    two_exposures.validate()
    assert len(dr.exposures_recorded()) == 2, "both exposures must remain in the history"
    assert dr.evidence_status() == "retrospective"


def test_the_submission_check_reads_the_committed_log_not_the_embedded_copy(two_exposures):
    """
    Deleting the append-only log left the check green, because it only ever read the list
    copied into replay.json - and a copy of a record is not evidence of the record.
    """
    # The artefact keeps the copy it was written with, as a real replay.json would;
    # only the committed log goes missing.
    embedded = dr.exposure_events()
    two_exposures.log.unlink()
    with pytest.raises(AssertionError, match="is missing"):
        two_exposures.validate(events=embedded)


# ---------------------------------------------------------------------------------------
# A SUMMARY IS CHECKED AGAINST WHAT IT SUMMARISES, NOT AGAINST ITSELF (recheck-5 S1)
# ---------------------------------------------------------------------------------------
# The check ran the shared classifier on the flags the REPORT supplied, so it asked only
# whether a report agreed with itself. Every mutation below leaves the frozen selection
# and the committed log exactly as the real producer wrote them, and changes ONE thing
# in replay.json's summary. Each used to pass.

def _relabelled(two_exposures, **changes):
    flags = dict(dr.evidence_flags())
    flags.update(changes)
    return flags


@pytest.mark.parametrize("field,value", [
    ("logged_exposures", 1),
    ("logged_exposures", 0),
    ("exposures_under_this_freeze", 2),
    ("exposures_under_earlier_freezes", 0),
    ("revalidated", False),
    ("prior_exposure_declared", True),
    ("exposures_declared_lost", 1),
    ("log_loss_declared", True),
    ("exposures_still_missing", 1),
    ("log_loss_in_effect", True),
    ("origin_declared", True),
])
def test_a_report_cannot_misstate_one_fact_about_its_own_evidence(two_exposures,
                                                                  field, value):
    real = dr.evidence_flags()
    assert real[field] != value, "the mutation must actually change the fact"
    flags = _relabelled(two_exposures, **{field: value})
    # Keep the report's status CONSISTENT with its false flags, so the only thing that
    # can catch this is comparison with the files - which is the point.
    flags["status"] = dr._classify_evidence(
        flags["logged_exposures"], flags["exposures_under_this_freeze"],
        flags["prior_exposure_declared"], flags["revalidated"], True,
        flags["log_loss_in_effect"])
    with pytest.raises(AssertionError, match="contradicts the evidence"):
        two_exposures.validate(flags=flags, status=flags["status"])


def test_a_false_prospective_label_is_rejected(two_exposures):
    """The reviewer's case: two real exposures relabelled as one clean prospective look."""
    flags = _relabelled(two_exposures, logged_exposures=1,
                        exposures_under_this_freeze=1,
                        exposures_under_earlier_freezes=0,
                        revalidated=False, status="prospective")
    with pytest.raises(AssertionError, match="contradicts the evidence"):
        two_exposures.validate(flags=flags, status="prospective")


def test_a_status_the_files_do_not_support_is_rejected_even_with_true_flags(two_exposures):
    with pytest.raises(AssertionError):
        two_exposures.validate(status="prospective")


_LATER_FLAGS = ("exposures_declared_lost", "log_loss_declared",
                "exposures_still_missing", "log_loss_in_effect", "origin_declared")


def test_a_report_written_before_the_lost_log_flags_existed_still_passes(two_exposures):
    """
    FOUND BY THE WORKED EXEMPLAR. Its committed replay.json predates the two lost-log
    flags, and the first version of the S1 check read their absence as a contradiction -
    so honest evidence produced a day earlier failed submission. A report that could not
    have recorded a declaration does not disagree with evidence that holds none.
    """
    flags = {k: v for k, v in dr.evidence_flags().items() if k not in _LATER_FLAGS}
    two_exposures.validate(flags=flags, status=flags["status"])


def test_omitting_the_lost_log_flags_cannot_hide_a_declared_loss(two_exposures):
    two_exposures.log.unlink()
    dr.freeze_selection(lost_log=True, note="synthetic: the log could not be restored")
    flags = {k: v for k, v in dr.evidence_flags().items() if k not in _LATER_FLAGS}
    with pytest.raises(AssertionError, match="contradicts the evidence"):
        two_exposures.validate(flags=flags, status=flags["status"])


def test_the_lost_log_route_ends_in_an_honest_submission_that_passes(two_exposures):
    """
    THE RECOVERY HAS TO FINISH. `--declare-lost-log` got the runtime unstuck, but the
    submission check still demanded a logged or pre-log exposure - and a lost log has
    neither - so a team that followed the supported route could never submit. It must
    pass, classified `previously_exposed`, with the benchmark bound to a covered exposure.
    """
    two_exposures.log.unlink()
    dr.freeze_selection(lost_log=True, note="synthetic: the log could not be restored")
    two_exposures.validate()
    assert dr.evidence_status() == "previously_exposed"


def test_a_lost_log_declaration_cannot_carry_a_benchmark_from_an_uncovered_exposure(
        two_exposures):
    two_exposures.log.unlink()
    dr.freeze_selection(lost_log=True, note="synthetic: the log could not be restored")
    with pytest.raises(AssertionError, match="declaration covers"):
        two_exposures.validate(event_id="an-exposure-nobody-recorded")


def test_a_report_cannot_drop_the_history_behind_a_prior_exposure_declaration(
        two_exposures, monkeypatch):
    """
    The second S1 case. The comparison with the committed log ran only when the EMBEDDED
    history held an open event - so replacing it with [] and zeroing the counts, under a
    genuine prior-exposure declaration, passed while the log recorded two exposures.
    """
    rec = dr._read_selection()
    rec["prior_exposure_declared"] = True
    rec["note"] = "synthetic: the holdout was run before the log existed"
    dr.SELECTION_STAMP.write_text(json.dumps(rec), encoding="utf-8")
    flags = _relabelled(two_exposures, logged_exposures=0,
                        exposures_under_this_freeze=0,
                        exposures_under_earlier_freezes=0,
                        prior_exposure_declared=True, status="previously_exposed")
    with pytest.raises(AssertionError, match="log records 2"):
        two_exposures.validate(events=[], flags=flags, status="previously_exposed",
                               event_id="none")


# ---- the controls: every legitimate route must still pass ----------------------------

def test_a_single_prospective_exposure_passes(monkeypatch, tmp_path):
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    monkeypatch.setattr(dr, "SELECTION_STAMP", outputs / "replay_selection.json")
    monkeypatch.setattr(dr, "EXPOSURE_LOG", outputs / "replay_exposures.jsonl")
    monkeypatch.setattr(dr, "dev_sample", lambda *a, **k: ["2020-01-01"])
    monkeypatch.setattr(dr, "holdout_sample", lambda *a, **k: ["2020-02-01"])
    dr.freeze_selection()
    event = dr.record_holdout_exposure("the one prospective look")
    dr.close_holdout_exposure(event["event_id"])
    sub = _submission_module()
    monkeypatch.setattr(sub, "ROOT", tmp_path)
    monkeypatch.setattr(sub, "_artefact", lambda name: {
        "selection": dr._read_selection(),
        "exposure": {"events": dr.exposure_events(),
                     "evidence_status": dr.evidence_status(),
                     "evidence_flags": dr.evidence_flags(),
                     "holdout_event_id": event["event_id"]}})
    sub.test_the_holdout_result_belongs_to_a_frozen_selection_and_a_recorded_exposure()
    assert dr.evidence_status() == "prospective"


def test_a_declared_legacy_exposure_with_no_log_passes(monkeypatch, tmp_path):
    """The documented no-log route: evidence that genuinely predates the log."""
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    monkeypatch.setattr(dr, "SELECTION_STAMP", outputs / "replay_selection.json")
    monkeypatch.setattr(dr, "EXPOSURE_LOG", outputs / "replay_exposures.jsonl")
    monkeypatch.setattr(dr, "dev_sample", lambda *a, **k: ["2020-01-01"])
    monkeypatch.setattr(dr, "holdout_sample", lambda *a, **k: ["2020-02-01"])
    dr.freeze_selection(prior_exposure=True,
                        note="synthetic: holdout run before the freeze workflow existed")
    assert not dr.EXPOSURE_LOG.exists()
    sub = _submission_module()
    monkeypatch.setattr(sub, "ROOT", tmp_path)
    monkeypatch.setattr(sub, "_artefact", lambda name: {
        "selection": dr._read_selection(),
        "exposure": {"events": [], "evidence_status": dr.evidence_status(),
                     "evidence_flags": dr.evidence_flags(),
                     "holdout_event_id": None}})
    sub.test_the_holdout_result_belongs_to_a_frozen_selection_and_a_recorded_exposure()
    assert dr.evidence_status() == "previously_exposed"


def test_a_same_configuration_revalidation_passes(monkeypatch, tmp_path):
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    monkeypatch.setattr(dr, "SELECTION_STAMP", outputs / "replay_selection.json")
    monkeypatch.setattr(dr, "EXPOSURE_LOG", outputs / "replay_exposures.jsonl")
    monkeypatch.setattr(dr, "dev_sample", lambda *a, **k: ["2020-01-01"])
    monkeypatch.setattr(dr, "holdout_sample", lambda *a, **k: ["2020-02-01"])
    dr.freeze_selection()
    a = dr.record_holdout_exposure("first")
    dr.close_holdout_exposure(a["event_id"])
    dr.freeze_selection(revalidate=True, note="synthetic: same configuration, second look")
    b = dr.record_holdout_exposure("second")
    dr.close_holdout_exposure(b["event_id"])
    sub = _submission_module()
    monkeypatch.setattr(sub, "ROOT", tmp_path)
    monkeypatch.setattr(sub, "_artefact", lambda name: {
        "selection": dr._read_selection(),
        "exposure": {"events": dr.exposure_events(),
                     "evidence_status": dr.evidence_status(),
                     "evidence_flags": dr.evidence_flags(),
                     "holdout_event_id": b["event_id"]}})
    sub.test_the_holdout_result_belongs_to_a_frozen_selection_and_a_recorded_exposure()
    assert dr.evidence_status() == "retrospective"


# ---------------------------------------------------------------------------------------
# THE RECOVERY A MISSING-LOG ERROR RECOMMENDS MUST ACTUALLY RECOVER (recheck-5 S5)
# ---------------------------------------------------------------------------------------

def _advertised_command(message: str) -> str:
    import re
    found = re.findall(r"python src/decision_replay\.py (--dev[^\n]*)", message)
    assert found, f"the error recommends no command at all:\n{message}"
    return found[-1]


@pytest.mark.parametrize("breakage", ["deleted", "damaged"])
def test_following_the_recovery_advice_recovers_rather_than_looping(two_exposures,
                                                                   breakage):
    """
    THE DEFECT THIS PINS. The missing-log error advised re-freezing with `--revalidate`,
    and `freeze_selection()` runs the same guard before changing anything - so the
    advertised command failed with the very error it was advertised to resolve. This
    takes the command FROM THE MESSAGE and runs what it names, so the advice and the
    code cannot drift apart again.
    """
    if breakage == "deleted":
        two_exposures.log.unlink()
    else:
        two_exposures.log.write_text("not json at all\n", encoding="utf-8")

    with pytest.raises(RuntimeError) as caught:
        dr.require_readable_log()
    command = _advertised_command(str(caught.value))
    assert "--declare-lost-log" in command, (
        f"the error must recommend the supported recovery, not {command!r}")

    counted_before = 2
    dr.freeze_selection(lost_log=True, note="synthetic: the log could not be restored")

    dr.require_readable_log()                          # no longer refuses
    assert dr.known_exposure_count() >= counted_before or breakage == "damaged", (
        "declaring a loss must not reduce the exposures counted against the selection")
    assert dr.evidence_status() == "previously_exposed"
    assert dr.evidence_flags()["log_loss_declared"] is True
    with pytest.raises(RuntimeError, match="authorises"):
        dr.record_holdout_exposure("an attempt to spend a look the declaration freed")


def test_the_recovery_route_makes_no_request(two_exposures, monkeypatch):
    """A recovery that spends quota to record a fact about the past is not a recovery."""
    monkeypatch.setattr(dr, "_prompts_written", lambda: True)
    monkeypatch.setattr(dr, "evaluate_all", lambda *a, **k: pytest.fail(
        "the lost-log declaration re-ran the development sample"))
    monkeypatch.setattr(dr, "client_", lambda *a, **k: pytest.fail(
        "the lost-log declaration made an API request"))
    two_exposures.log.unlink()
    dr.run_dev(lost_log=True, note="synthetic: the log could not be restored")
    assert dr.evidence_flags()["exposures_declared_lost"] == 2


def test_a_lost_log_cannot_be_declared_when_nothing_is_lost(two_exposures):
    with pytest.raises(RuntimeError, match="nothing to declare"):
        dr.freeze_selection(lost_log=True, note="synthetic")


def test_a_lost_log_declaration_needs_a_note(two_exposures):
    two_exposures.log.unlink()
    with pytest.raises(RuntimeError, match="say what happened"):
        dr.freeze_selection(lost_log=True)


def test_an_exposure_is_stamped_into_the_freeze_when_it_is_recorded(monkeypatch, tmp_path):
    """
    Found while fixing S5. The freeze only learned an exposure's id at the NEXT freeze, so
    deleting the log between recording an exposure and re-freezing - that is, during the
    whole holdout run - erased it with nothing to notice.
    """
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    monkeypatch.setattr(dr, "SELECTION_STAMP", outputs / "replay_selection.json")
    monkeypatch.setattr(dr, "EXPOSURE_LOG", outputs / "replay_exposures.jsonl")
    monkeypatch.setattr(dr, "dev_sample", lambda *a, **k: ["2020-01-01"])
    monkeypatch.setattr(dr, "holdout_sample", lambda *a, **k: ["2020-02-01"])
    dr.freeze_selection()
    event = dr.record_holdout_exposure("recorded, never re-frozen")
    assert event["event_id"] in dr._read_selection()["exposure_event_ids"]
    dr.EXPOSURE_LOG.unlink()
    with pytest.raises(RuntimeError, match="no longer contains"):
        dr.require_readable_log()


@pytest.mark.parametrize("damage,label", [
    ("not json at all", "a line that is not JSON"),
    ("42", "a JSON scalar where an event belongs"),
])
def test_a_damaged_log_is_rejected_rather_than_read_as_a_shorter_history(two_exposures,
                                                                        damage, label):
    original = two_exposures.log.read_text(encoding="utf-8")
    two_exposures.log.write_text(original + damage + "\n", encoding="utf-8")
    with pytest.raises(AssertionError, match="damaged"):
        two_exposures.validate()
    two_exposures.log.write_text(original, encoding="utf-8")
    two_exposures.validate()          # control restored


def test_an_event_missing_a_required_field_is_damage(two_exposures):
    lines = two_exposures.log.read_text(encoding="utf-8").splitlines()
    stripped = json.loads(lines[0])
    stripped.pop("freeze_id")
    two_exposures.log.write_text(
        "\n".join([json.dumps(stripped)] + lines[1:]) + "\n", encoding="utf-8")
    with pytest.raises(AssertionError, match="damaged"):
        two_exposures.validate()


def test_the_reported_benchmark_must_name_an_event_of_the_current_freeze(two_exposures):
    with pytest.raises(AssertionError, match="names"):
        two_exposures.validate(event_id="deadbeef1234")
    # A's event is real and legitimate HISTORY, but it is not what this report rests on.
    with pytest.raises(AssertionError, match="belongs to exposure"):
        two_exposures.validate(event_id=two_exposures.a["event_id"])
    two_exposures.validate()          # control restored


def test_a_substituted_history_is_rejected(two_exposures):
    only_b = [e for e in dr.exposure_events()
              if e.get("event_id") == two_exposures.b["event_id"]]
    with pytest.raises(AssertionError, match="not the history"):
        two_exposures.validate(events=only_b)


def test_the_runtime_refuses_to_count_a_damaged_history(two_exposures):
    """
    A count derived from a damaged log is not a count. Reading the damage as "zero
    exposures so far" would have bought another look at the held-out sample.
    """
    original = two_exposures.log.read_text(encoding="utf-8")
    two_exposures.log.write_text(original + "malformed event\n", encoding="utf-8")
    for call in (dr._require_frozen_selection, dr.record_holdout_exposure,
                 dr.freeze_selection):
        with pytest.raises(RuntimeError, match="damaged"):
            call()
    two_exposures.log.write_text(original, encoding="utf-8")
    dr._require_frozen_selection()    # control restored


def test_a_cache_only_replay_keeps_the_exposure_its_evidence_came_from(two_exposures):
    """
    Re-running the stage from committed caches asks the holdout nothing, so it records no
    new event - but the numbers it reports still belong to the exposure that produced
    them. Writing `null` there severed the report from its own evidence.
    """
    assert dr.originating_exposure_id() == two_exposures.b["event_id"]
    assert dr.originating_exposure_id() is not None


def test_an_unclosed_exposure_is_not_a_finished_benchmark(cached_holdout):
    """A benchmark that never finished - its exposure opened and never closed - is no result."""
    dr.freeze_selection()
    event = dr.record_holdout_exposure("a look that never finished")
    report = {"selection": dr._read_selection(),
              "exposure": {"events": dr.exposure_events(),
                           "evidence_status": dr.evidence_status(),
                           "evidence_flags": dr.evidence_flags(),
                           "holdout_event_id": event["event_id"]}}
    with pytest.raises(AssertionError, match="never closed"):
        cached_holdout.submit(report)


def test_a_finished_benchmark_whose_close_was_deleted_is_a_missing_record(two_exposures):
    """
    (recheck-7 U1) Deleting a finished benchmark's close line used to leave an exposure every
    check read as merely unclosed - and an unclosed exposure of the current freeze is one a
    run resumes, with new requests. The freeze now records the completion, so the deletion is
    a MISSING RECORD: refused by the runtime and at submission until it is restored or
    declared, and the benchmark stays finished either way.
    """
    lines = [json.loads(x) for x in
             two_exposures.log.read_text(encoding="utf-8").splitlines()]
    kept = [x for x in lines
            if not (x.get("type") == "close"
                    and x.get("event_id") == two_exposures.b["event_id"])]
    two_exposures.log.write_text(
        "\n".join(json.dumps(x) for x in kept) + "\n", encoding="utf-8")
    with pytest.raises(AssertionError, match="record of it finishing"):
        two_exposures.validate()
    with pytest.raises(RuntimeError, match="record of it finishing"):
        dr.require_readable_log()
    assert dr.open_exposure(two_exposures.b["freeze_id"]) is None, (
        "a benchmark the freeze recorded as finished must not be resumable")

    dr.freeze_selection(lost_log=True, note="synthetic: the log could not be restored")
    dr.require_readable_log()
    assert dr.open_exposure(two_exposures.b["freeze_id"]) is None
    with pytest.raises(AssertionError, match="never closed"):
        two_exposures.validate()
    dr.close_holdout_exposure(two_exposures.b["event_id"],
                              "benchmark completed (replayed from committed cache)")
    two_exposures.validate()


# -------------------------------------------------------------------------------------------
# THE EXPOSURE IS DURABLE BEFORE ANY WORKER REQUESTS HOLDOUT DATA  (recheck-3, plan T7)
# -------------------------------------------------------------------------------------------

def _holdout_workers(monkeypatch, tmp_path, append=None, workers=8):
    """Start `workers` threads at a barrier, all entering the holdout path together.

    A BARRIER, NOT A SLEEP. Timing-dependent tests pass on a fast machine for the wrong
    reason; the barrier makes every worker arrive at the same instant on every machine.
    """
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    monkeypatch.setattr(dr, "SELECTION_STAMP", outputs / "replay_selection.json")
    monkeypatch.setattr(dr, "EXPOSURE_LOG", outputs / "replay_exposures.jsonl")
    monkeypatch.setattr(dr, "dev_sample", lambda *a, **k: ["2020-01-01"])
    monkeypatch.setattr(dr, "holdout_sample", lambda *a, **k: ["2020-02-01"])
    dr.freeze_selection()
    if append is not None:
        monkeypatch.setattr(dr, "_append_event", append)

    monkeypatch.setattr(dr, "_CURRENT_SAMPLE", "holdout")
    monkeypatch.setattr(dr, "_EXPOSURE_RECORDED", False)
    monkeypatch.setattr(dr, "_EXPOSURE_EVENT", None)

    barrier = threading.Barrier(workers)
    observed, errors = [], []

    def worker():
        barrier.wait()
        try:
            with dr._EXPOSURE_LOCK:
                if not dr._EXPOSURE_RECORDED:
                    dr._EXPOSURE_EVENT = dr.record_holdout_exposure("concurrent")
                    dr._EXPOSURE_RECORDED = True
            # Standing in for "this worker now makes its first request": what the log
            # holds at this instant is what the request would be covered by.
            observed.append(len(dr.exposures_recorded()))
        except Exception as exc:                         # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return observed, errors


def test_exactly_one_exposure_is_durable_before_any_worker_proceeds(monkeypatch,
                                                                    tmp_path):
    observed, errors = _holdout_workers(monkeypatch, tmp_path)
    assert not errors, errors
    assert observed and set(observed) == {1}, (
        f"workers saw {sorted(set(observed))} recorded exposures; every one of them must "
        f"see exactly the single durable event before it would make a request")
    assert len(dr.exposures_recorded()) == 1


def test_a_failed_append_lets_no_worker_reach_transport(monkeypatch, tmp_path):
    """
    If the event cannot be made durable, nothing may proceed as though it had been. The
    flag must not be set, and the failure must surface rather than being swallowed.
    """
    def broken_append(event):
        raise OSError("synthetic: the exposure log could not be written")

    observed, errors = _holdout_workers(monkeypatch, tmp_path, append=broken_append)
    assert not observed, (
        "a worker carried on to make a request after the exposure record failed to write")
    assert errors and all(isinstance(e, OSError) for e in errors), errors
    assert dr._EXPOSURE_RECORDED is False, (
        "a failed append must not leave the run believing the exposure was recorded")
    assert not dr.exposures_recorded()


# -------------------------------------------------------------------------------------------
# FILTERED DRAWS: TOLERATED VS BLOCKING  (recheck-3 N6, plan T10)
# -------------------------------------------------------------------------------------------

def _parsed_payload():
    import text_features as tf
    payload = {}
    for field in tf.FIELDS:
        payload[field] = 5.0
        if tf.REQUIRE_EVIDENCE:
            payload[f"{field}_evidence"] = "the Board judged"
    return payload


def _filtered_draw(idx):
    return {"ok": False, "call_index": config.CALL_INDEX_BASE + idx,
            "error": "ContentFilteredError: synthetic",
            "failure": {"category": "content_filter", "settled": True,
                        "run_level": False, "error_type": "ContentFilteredError"}}


@pytest.mark.parametrize("n_valid", [config.MIN_VALID_CALLS,
                                     config.N_PARALLEL_CALLS - 1])
def test_filtered_draws_are_tolerated_while_enough_valid_ones_remain(n_valid):
    """
    The rubric's first filter row promises the document still scores. It only does while
    `MIN_VALID_CALLS` draws came back, which is exactly the distinction the rubric used
    to leave out.
    """
    import pandas as pd
    import text_features as tf

    calls = [{"ok": True, "call_index": config.CALL_INDEX_BASE + i,
              "parsed": _parsed_payload()} for i in range(n_valid)]
    calls += [_filtered_draw(i) for i in range(n_valid, config.N_PARALLEL_CALLS)]
    docs = pd.DataFrame([{"meeting_date": "2020-01-01",
                          "text_scored": "the Board judged that conditions warranted"}])
    df, _ev = tf.aggregate([{"meeting_date": "2020-01-01", "calls": calls}], docs)
    assert len(df) == 1
    assert int(df.iloc[0]["n_calls_valid"]) == n_valid


def test_a_document_filtered_below_the_minimum_stops_the_stage_with_usable_advice():
    """
    Below the minimum the stage must stop rather than average what is left - and say
    something a team can act on. "Re-run" is the wrong advice: a filtered draw is settled,
    so an unchanged re-run makes no calls at all and the loop cannot terminate.
    """
    import pandas as pd
    import text_features as tf

    calls = [_filtered_draw(i) for i in range(config.N_PARALLEL_CALLS)]
    docs = pd.DataFrame([{"meeting_date": "2020-01-01", "text_scored": "synthetic"}])
    with pytest.raises(RuntimeError) as caught:
        tf.aggregate([{"meeting_date": "2020-01-01", "calls": calls}], docs)
    message = str(caught.value)
    assert "content_filter" in message, "the message must say WHY the draws failed"
    assert "will not clear them" in message
    assert "DO NOT edit the corpus" in message
    assert "email the lecturer" in message


def test_a_truncated_document_below_the_minimum_is_told_to_raise_the_ceiling():
    """A different cause needs different advice; one generic message served neither."""
    import pandas as pd
    import text_features as tf

    calls = [dict(_filtered_draw(i),
                  failure={"category": "truncated", "settled": True, "run_level": False,
                           "error_type": "IncompleteResponseError"})
             for i in range(config.N_PARALLEL_CALLS)]
    docs = pd.DataFrame([{"meeting_date": "2020-01-01", "text_scored": "synthetic"}])
    with pytest.raises(RuntimeError, match="MAX_OUTPUT_TOKENS"):
        tf.aggregate([{"meeting_date": "2020-01-01", "calls": calls}], docs)


# -------------------------------------------------------------------------------------------
# MIXED FAILURES: RECOVERABLE VS GENUINELY BLOCKED  (recheck-4 R5)
# -------------------------------------------------------------------------------------------

def _mixed(valid, **failures):
    """`valid` successful draws plus counts of failures by category."""
    calls = [{"ok": True, "call_index": config.CALL_INDEX_BASE + i,
              "parsed": _parsed_payload()} for i in range(valid)]
    idx = valid
    for category, n in failures.items():
        for _ in range(n):
            calls.append(dict(_filtered_draw(idx),
                              failure={"category": category,
                                       "settled": category != "transport",
                                       "run_level": False,
                                       "error_type": "Synthetic"},
                              error=f"{category}: synthetic"))
            idx += 1
    return calls


def _aggregate(calls):
    import pandas as pd
    import text_features as tf
    docs = pd.DataFrame([{"meeting_date": "2020-01-01",
                          "text_scored": "the Board judged that conditions warranted"}])
    return tf.aggregate([{"meeting_date": "2020-01-01", "calls": calls}], docs)


@pytest.mark.parametrize("calls,expect", [
    # RECOVERABLE: the ceiling can still lift these over the minimum.
    (_mixed(2, content_filter=1, truncated=2), "MAX_OUTPUT_TOKENS"),
    # RECOVERABLE: a retry can.
    (_mixed(2, content_filter=1, transport=2), "failed in transit"),
    (_mixed(0, truncated=5), "MAX_OUTPUT_TOKENS"),
])
def test_a_recoverable_document_is_told_how_to_recover(calls, expect):
    """
    THE REGRESSION THIS PINS. The blocked/recoverable test compared the filtered count
    against `N_PARALLEL_CALLS - len(ok)`, which double-counts the valid draws: two valid
    draws, one filtered and two TRUNCATED ones can still reach three valid, and were
    nevertheless declared blocked and sent to the service-failure waiver. Raising the
    ceiling would have recovered them.
    """
    with pytest.raises(RuntimeError) as caught:
        _aggregate(calls)
    message = str(caught.value)
    assert "can still reach the minimum" in message, message
    assert expect in message, message
    assert "will not clear them" not in message, (
        "a recoverable document must not be sent to the approved-service waiver")


@pytest.mark.parametrize("calls", [
    _mixed(0, content_filter=5),
    _mixed(2, content_filter=3),
    _mixed(1, content_filter=3, refusal=1),
])
def test_a_genuinely_blocked_document_is_told_so(calls):
    """Below the minimum with nothing recoverable left: the waiver route, and no loop."""
    with pytest.raises(RuntimeError) as caught:
        _aggregate(calls)
    message = str(caught.value)
    assert "will not clear them" in message, message
    assert "DO NOT edit the corpus" in message
    assert "email the lecturer" in message
    assert "can still reach the minimum" not in message


@pytest.mark.parametrize("valid", [config.MIN_VALID_CALLS,
                                   config.N_PARALLEL_CALLS - 1])
def test_enough_valid_draws_aggregate_despite_filtered_ones(valid):
    df, _ev = _aggregate(_mixed(valid,
                                content_filter=config.N_PARALLEL_CALLS - valid))
    assert int(df.iloc[0]["n_calls_valid"]) == valid


# -------------------------------------------------------------------------------------------
# THE DOCUMENTED SUBMISSION ROUTES ARE THE ONES THE CHECKER ACCEPTS  (recheck-4 R7)
# -------------------------------------------------------------------------------------------

@pytest.mark.parametrize("layout,accepted", [
    ("paired", True),          # what docs/README.md now tells students to do
    ("combined", False),       # what it used to tell them to do
    ("moodle", True),          # the declared restricted route
    ("subfolder", True),       # the documented per-date folder
])
def test_each_documented_meeting_layout_matches_the_checker(tmp_path, monkeypatch,
                                                            layout, accepted):
    """
    THE DRIFT THIS PINS. The published `docs/README.md` and `docs/meetings/README.md` told
    students to put the transcript and minutes together in ONE file per meeting. The
    checker requires separately named files for each date, so a team following the
    instruction in the repository they were given would fail the completeness check.
    """
    sub = _submission_module()
    monkeypatch.setattr(sub, "ROOT", tmp_path)
    meetings = tmp_path / "docs" / "meetings"
    meetings.mkdir(parents=True)

    if layout == "paired":
        for day in ("2026-03-16", "2026-03-23"):
            for kind in ("transcript", "minutes"):
                (meetings / f"{day}-{kind}.md").write_text("record", encoding="utf-8")
    elif layout == "combined":
        for day in ("2026-03-16", "2026-03-23"):
            (meetings / f"{day}-meeting.md").write_text(
                "transcript and minutes together", encoding="utf-8")
    elif layout == "subfolder":
        for day in ("2026-03-16", "2026-03-23"):
            (meetings / day).mkdir()
            for kind in ("transcript", "minutes"):
                (meetings / day / f"{kind}.md").write_text("record", encoding="utf-8")
    else:
        (meetings / "README.md").write_text("SUBMISSION ROUTE: Moodle\n",
                                            encoding="utf-8")

    if accepted:
        sub.test_meeting_records_are_paired_transcripts_and_minutes()
    else:
        with pytest.raises(AssertionError):
            sub.test_meeting_records_are_paired_transcripts_and_minutes()


@pytest.mark.parametrize("name", ["cycle-transcript.md", "cycle-transcript.pdf"])
def test_the_cycle_transcript_gate_accepts_a_file_holding_a_share_link(tmp_path,
                                                                      monkeypatch, name):
    """The documented route - export OR share link - must satisfy the gate as a FILE."""
    sub = _submission_module()
    monkeypatch.setattr(sub, "ROOT", tmp_path)
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / name).write_text("https://chatgpt.com/share/synthetic-link\n",
                             encoding="utf-8")
    sub.test_the_cycle_transcript_is_in_the_repository()


def test_the_shipped_signposts_do_not_satisfy_any_gate(tmp_path, monkeypatch):
    """
    THE BUG THIS PINS. `docs/meetings/README.md` ships as a signpost explaining both
    routes, so it necessarily mentions Moodle, transcripts and minutes - and it SHOWED the
    declaration line as an example. Both facts made the untouched starter pass the meeting
    gate: the prose fallback read the explanation as a choice, and the regex read the
    example as the choice itself. A template must declare nothing.
    """
    import shutil
    sub = _submission_module()
    monkeypatch.setattr(sub, "ROOT", tmp_path)
    src = pathlib.Path(__file__).resolve().parent.parent / "docs"
    shutil.copytree(src, tmp_path / "docs")
    assert (tmp_path / "docs/meetings/README.md").exists(), "fixture did not copy"

    assert sub._declared_route(tmp_path / "docs/meetings/README.md") is None, (
        "the shipped signpost declares a submission route; as shipped it must declare "
        "nothing, or an untouched starter passes the meeting gate by doing nothing")
    with pytest.raises(AssertionError):
        sub.test_meeting_records_are_paired_transcripts_and_minutes()

    # And a team that fills it in IS a declaration.
    (tmp_path / "docs/meetings/README.md").write_text(
        "SUBMISSION ROUTE: Moodle\n", encoding="utf-8")
    sub.test_meeting_records_are_paired_transcripts_and_minutes()


# -------------------------------------------------------------------------------------------
# THE LOST-LOG ROUTE, FROM THE PRODUCER TO SUBMISSION  (recheck-6 T1, T2, T3)
# -------------------------------------------------------------------------------------------
# Every case runs the REAL cached benchmark producer after the damage, and passes what it
# emits - event id, history and flags, unchanged - to the submission check. The earlier
# route test supplied a fixture's saved event id: exactly the thing the producer lost.

_CACHED_RECOMMENDATION = {"ok": True,
                          "result": {"recommendation": "hold", "size_bp": 0,
                                     "confidence": 0.8},
                          "shot_mix": {"hold_share": 1.0}}


@pytest.fixture()
def cached_holdout(monkeypatch, tmp_path):
    """A frozen workspace whose holdout benchmark is served entirely from cache."""
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    monkeypatch.setattr(dr, "SELECTION_STAMP", outputs / "replay_selection.json")
    monkeypatch.setattr(dr, "EXPOSURE_LOG", outputs / "replay_exposures.jsonl")
    monkeypatch.setattr(dr, "dev_sample", lambda *a, **k: ["2020-01-01"])
    monkeypatch.setattr(dr, "holdout_sample", lambda *a, **k: ["2020-02-01"])
    monkeypatch.setattr(dr, "_prompts_written", lambda: True)
    requests = {"n": 0}

    def no_network(*a, **k):
        requests["n"] += 1
        raise RuntimeError("a cached benchmark must not make a request")

    monkeypatch.setattr(dr, "client_", no_network)
    monkeypatch.setattr(dr, "recommend", lambda *a, **k: dict(_CACHED_RECOMMENDATION))
    monkeypatch.setattr(dr, "feasible_everywhere",
                        lambda meetings, strategies, k: (meetings, {}))
    monkeypatch.setattr(dr, "actual_decision",
                        lambda *a, **k: {"word": "hold", "size_bp": 0, "decision": 0})
    sub = _submission_module()
    monkeypatch.setattr(sub, "ROOT", tmp_path)

    def benchmark():
        return dr.evaluate_all(meetings=["2020-02-01"], strategies=["recent"], n_seeds=1,
                               max_workers=1, sample="holdout")

    def report(df):
        return {"selection": dr._read_selection(),
                "exposure": {"events": dr.exposure_events(),
                             "evidence_status": dr.evidence_status(),
                             "evidence_flags": dr.evidence_flags(),
                             "holdout_event_id": df.attrs["exposure_event_id"]}}

    def submit(rep):
        monkeypatch.setattr(sub, "_artefact", lambda name: rep)
        sub.test_the_holdout_result_belongs_to_a_frozen_selection_and_a_recorded_exposure()

    def expose(reason):
        event = dr.record_holdout_exposure(reason)
        dr.close_holdout_exposure(event["event_id"])
        return event["event_id"]

    def lines():
        return dr.EXPOSURE_LOG.read_text(encoding="utf-8").splitlines()

    def write(kept, *, newline_at_end=True):
        body = "\n".join(kept) + ("\n" if newline_at_end else "")
        dr.EXPOSURE_LOG.write_bytes(body.encode("utf-8"))

    return types.SimpleNamespace(benchmark=benchmark, report=report, submit=submit,
                                 expose=expose, lines=lines, write=write,
                                 requests=requests)


def _without_event(lines, event_id):
    return [line for line in lines if json.loads(line).get("event_id") != event_id]


def _declare(note="synthetic: the log could not be restored from version control"):
    dr.run_dev(lost_log=True, note=note)


def test_after_a_complete_loss_the_cached_producer_keeps_the_benchmark_origin(
        cached_holdout):
    """
    THE DEFECT THIS PINS (T1). The origin came from the readable log alone, so after a
    declared loss the cached producer emitted `exposure_event_id: null` and submission
    refused the recovery the error message had recommended.
    """
    w = cached_holdout
    dr.freeze_selection()
    origin = w.expose("the one look at the holdout")
    before = w.benchmark()
    assert before.attrs["exposure_event_id"] == origin
    authorised = dr._read_selection()["authorised_exposures"]

    dr.EXPOSURE_LOG.unlink()
    _declare()
    after = w.benchmark()

    assert after.attrs["exposure_event_id"] == origin, (
        "the cached producer lost the benchmark's originating exposure")
    assert after.to_dict("records") == before.to_dict("records"), (
        "recovery must not change the numbers the benchmark reports")
    assert dr._read_selection()["authorised_exposures"] == authorised
    assert dr.known_exposure_count() == 1
    assert w.requests["n"] == 0
    w.submit(w.report(after))
    assert dr.evidence_status() == "previously_exposed"


@pytest.mark.parametrize("lost", ["the later exposure", "the earlier exposure"])
def test_a_partial_loss_never_moves_the_benchmark_to_another_exposure(cached_holdout, lost):
    """
    THE DEFECT THIS PINS (T1). With two exposures under one freeze, losing the later one's
    records made the producer attach the benchmark to the EARLIER one - the last survivor -
    and submission accepted it, while the honestly preserved id was refused because any
    readable event sent the check down the readable-log branch.
    """
    w = cached_holdout
    dr.freeze_selection()
    first = w.expose("the first look")
    dr.freeze_selection(revalidate=True, note="synthetic: a declared second look")
    second = w.expose("the second look")
    before = w.benchmark()
    assert before.attrs["exposure_event_id"] == second

    gone = second if lost == "the later exposure" else first
    w.write(_without_event(w.lines(), gone))
    _declare()
    after = w.benchmark()

    assert after.attrs["exposure_event_id"] == second, (
        f"after losing {lost}, the producer bound the benchmark to "
        f"{after.attrs['exposure_event_id']!r} instead of the latest exposure {second!r}")
    assert after.to_dict("records") == before.to_dict("records")
    assert dr.known_exposure_count() == 2
    assert w.requests["n"] == 0
    rep = w.report(after)
    w.submit(rep)

    older = json.loads(json.dumps(rep))
    older["exposure"]["holdout_event_id"] = first
    with pytest.raises(AssertionError, match="latest exposure"):
        w.submit(older)


def test_a_partial_loss_under_a_changed_freeze_keeps_the_current_freezes_exposure(
        cached_holdout, monkeypatch):
    w = cached_holdout
    dr.freeze_selection()
    first = w.expose("a look under the first selection")
    monkeypatch.setattr(config, "CALL_INDEX_BASE", config.CALL_INDEX_BASE + 100)
    dr.freeze_selection(revalidate=True, note="synthetic: re-selected and declared")
    second = w.expose("a look under the new selection")
    before = w.benchmark()

    w.write(_without_event(w.lines(), second))
    _declare()
    after = w.benchmark()

    assert after.attrs["exposure_event_id"] == second
    assert after.to_dict("records") == before.to_dict("records")
    rep = w.report(after)
    w.submit(rep)
    older = json.loads(json.dumps(rep))
    older["exposure"]["holdout_event_id"] = first
    with pytest.raises(AssertionError):
        w.submit(older)


_DAMAGE = ("malformed JSON", "truncated final line", "incomplete open event",
           "corrupt close event")


@pytest.mark.parametrize("damage", _DAMAGE)
def test_declared_damage_is_accepted_end_to_end_and_later_damage_is_not(cached_holdout,
                                                                        damage):
    """
    THE DEFECT THIS PINS (T2). A declared corruption passed the runtime guard but failed
    submission unconditionally, which told the team to restore the file it had just
    declared unrecoverable. And the runtime waived EVERY damaged line once any declaration
    existed, so corruption added afterwards was accepted too.
    """
    w = cached_holdout
    dr.freeze_selection()
    origin = w.expose("the one look at the holdout")
    before = w.benchmark()
    lines = w.lines()
    open_at = next(i for i, l in enumerate(lines) if json.loads(l).get("type") == "open")
    close_at = next(i for i, l in enumerate(lines) if json.loads(l).get("type") == "close")
    newline_at_end = True
    if damage == "malformed JSON":
        lines[open_at] = "{synthetic damage"
    elif damage == "truncated final line":
        lines[-1] = lines[-1][: len(lines[-1]) // 2]
        newline_at_end = False
    elif damage == "incomplete open event":
        event = json.loads(lines[open_at])
        event.pop("freeze_id")
        lines[open_at] = json.dumps(event, sort_keys=True)
    else:
        event = json.loads(lines[close_at])
        event.pop("event_id")
        lines[close_at] = json.dumps(event, sort_keys=True)
    w.write(lines, newline_at_end=newline_at_end)
    damaged_bytes = dr.EXPOSURE_LOG.read_bytes()

    with pytest.raises(RuntimeError, match="damaged"):
        dr.require_readable_log()
    _declare()
    dr.require_readable_log()
    after = w.benchmark()

    assert after.attrs["exposure_event_id"] == origin
    assert after.to_dict("records") == before.to_dict("records")
    assert dr.known_exposure_count() == 1
    assert dr._read_selection()["authorised_exposures"] == 1
    assert w.requests["n"] == 0
    assert dr.EXPOSURE_LOG.read_bytes().startswith(damaged_bytes), (
        "the damaged history must be kept byte for byte - never deleted or rebuilt")
    w.submit(w.report(after))

    with dr.EXPOSURE_LOG.open("a", encoding="utf-8") as fh:
        fh.write("{damage introduced after the declaration\n")
    with pytest.raises(RuntimeError, match="damaged"):
        dr.require_readable_log()
    with pytest.raises(AssertionError, match="damaged"):
        w.submit(w.report(after))


@pytest.mark.parametrize("restored", ["every record", "every record twice",
                                      "the open event only"])
def test_restored_records_are_the_same_exposure_not_a_second_one(cached_holdout, restored):
    """
    THE DEFECT THIS PINS (T3). The count ADDED readable events to declared-lost ids, so
    restoring a lost exposure's own records counted it twice; submission then refused the
    restored evidence against its unchanged allowance and asked for --revalidate.
    """
    w = cached_holdout
    dr.freeze_selection()
    origin = w.expose("the one look at the holdout")
    original = w.lines()
    dr.EXPOSURE_LOG.unlink()
    _declare()
    declaration = w.lines()

    back = original if restored != "the open event only" else [
        l for l in original if json.loads(l).get("type") == "open"]
    w.write(back * (2 if restored == "every record twice" else 1) + declaration)

    assert dr.known_exposure_count() == 1
    dr.require_readable_log()
    after = w.benchmark()
    assert after.attrs["exposure_event_id"] == origin
    assert w.requests["n"] == 0
    w.submit(w.report(after))
    flags = dr.evidence_flags()
    assert flags["log_loss_declared"] is True, "the declaration stays on record as history"
    assert flags["exposures_still_missing"] == 0
    assert flags["log_loss_in_effect"] is False
    assert dr.evidence_status() == "prospective"

    # A GENUINELY new exposure is still a new look, and still needs declared authority.
    with pytest.raises(RuntimeError, match="authorises"):
        dr.record_holdout_exposure("an undeclared second look")
    dr.freeze_selection(revalidate=True, note="synthetic: a declared second look")
    newer = w.expose("the declared second look")
    assert dr.known_exposure_count() == 2
    latest = w.benchmark()
    assert latest.attrs["exposure_event_id"] == newer
    w.submit(w.report(latest))


def test_restoring_some_of_several_lost_exposures_keeps_the_rest_declared(cached_holdout):
    """
    Two exposures declared lost, one restored: the count stays at two, the loss stays in
    effect for the one still missing, and the benchmark stays bound to the latest exposure
    while it is still gone. Restoring the other ends the loss without adding a look.
    """
    w = cached_holdout
    dr.freeze_selection()
    first = w.expose("the first look")
    dr.freeze_selection(revalidate=True, note="synthetic: a declared second look")
    second = w.expose("the second look")
    original = w.lines()
    dr.EXPOSURE_LOG.unlink()
    _declare()
    declaration = w.lines()

    def records_of(event_id):
        return [line for line in original if json.loads(line).get("event_id") == event_id]

    w.write(records_of(first) + declaration)
    assert dr.known_exposure_count() == 2
    flags = dr.evidence_flags()
    assert flags["exposures_declared_lost"] == 2
    assert flags["exposures_still_missing"] == 1
    assert flags["log_loss_in_effect"] is True
    assert dr.evidence_status() == "previously_exposed"
    after = w.benchmark()
    assert after.attrs["exposure_event_id"] == second
    w.submit(w.report(after))

    w.write(w.lines() + records_of(second))
    assert dr.known_exposure_count() == 2
    flags = dr.evidence_flags()
    assert flags["exposures_still_missing"] == 0
    assert flags["log_loss_in_effect"] is False
    assert dr.evidence_status() == "retrospective"
    assert dr._read_selection()["authorised_exposures"] == 2
    latest = w.benchmark()
    assert latest.attrs["exposure_event_id"] == second
    w.submit(w.report(latest))
    assert w.requests["n"] == 0


def test_a_client_on_some_other_transport_still_writes_records_with_their_requests(
        tmp_path, monkeypatch):
    """
    Counting at dispatch relies on the transport telling the counters. A client built on a
    plain transport - the way a review probe or a test double builds one - has no token
    guards and refuses nothing, so each request it carries must still be counted. Otherwise
    a real success is written as having taken no request, and refused on its next reload.
    """
    import warnings
    import httpx
    import openai
    from pydantic import BaseModel
    import courseapi
    import unsw_ai

    class CacheProbe(BaseModel):
        value: int

    sent = []

    def network(request):
        sent.append(request)
        return httpx.Response(200, json={
            "id": "resp_synthetic", "object": "response", "created_at": 1,
            "model": config.MODEL, "status": "completed", "incomplete_details": None,
            "output": [{"type": "function_call", "id": "f", "call_id": "c",
                        "name": "CacheProbe", "arguments": '{"value":1}',
                        "status": "completed"}],
            "usage": {"input_tokens": 10, "output_tokens": 10, "total_tokens": 20}})

    sdk = openai.OpenAI(api_key="synthetic", base_url="https://synthetic.invalid",
                        max_retries=0,
                        http_client=httpx.Client(transport=httpx.MockTransport(network)))
    settings = unsw_ai.ProxySettings(proxy_url="https://synthetic.invalid",
                                     access_code="synthetic", student_id="9999999",
                                     fallback_models=())
    monkeypatch.setattr(unsw_ai, "build_openai_client", lambda *a, **k: sdk)
    monkeypatch.setattr(unsw_ai.time, "sleep", lambda *a, **k: None)
    monkeypatch.setattr(config, "ledger_add", lambda *a, **k: None)
    monkeypatch.setattr(dr, "RAW_DIR", tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        client = unsw_ai.UNSWInstructor(settings=settings)
        monkeypatch.setattr(unsw_ai, "get_client", lambda *a, **k: client)
        monkeypatch.setattr(dr, "client_", lambda *a, **k: courseapi.CourseClient())
        dr._cached_call("probe", "system", "user", schema=CacheProbe)
    sdk.close()
    rec = json.loads(next(tmp_path.glob("*.json")).read_text(encoding="utf-8"))
    assert rec["transport_requests"] == len(sent) == 1
    assert courseapi.envelope_contradictions(rec) == []
    assert client.usage.summary()["requests"] == len(sent)


# -------------------------------------------------------------------------------------------
# A REQUEST IS COUNTED WHERE IT IS DISPATCHED - envelopes AND session totals  (recheck-6 T4)
# -------------------------------------------------------------------------------------------
# Only the network is scripted. The wrapper's ProxyTransport and its token guard run for real,
# so a request the client refuses itself can be told apart from one it sends - the one thing
# the earlier harnesses, which replaced the whole transport, could never see. Requests were
# counted before those guards, and a prompt over the per-request cap wrote an envelope saying
# one real request of unknown cost had been made.

_T4_OUTCOMES = ("success", "HTTP 400", "HTTP 429", "timeout", "500 then success",
                "schema repaired", "schema never satisfied", "per-request cap",
                "daily budget", "a parameter the adapter refuses")
_T4_REFUSED_BEFORE_SENDING = ("per-request cap", "daily budget",
                              "a parameter the adapter refuses")


def _t4_responder(outcome, structured):
    import httpx

    def answer(arguments='{"value":1}'):
        if structured:
            output = [{"type": "function_call", "id": "f", "call_id": "c",
                       "name": "DispatchProbe", "arguments": arguments,
                       "status": "completed"}]
        else:
            output = [{"type": "message", "id": "m", "role": "assistant",
                       "status": "completed",
                       "content": [{"type": "output_text", "text": "synthetic",
                                    "annotations": []}]}]
        return httpx.Response(200, json={
            "id": "resp_synthetic", "object": "response", "created_at": 1,
            "model": config.MODEL, "status": "completed", "incomplete_details": None,
            "output": output,
            "usage": {"input_tokens": 10, "output_tokens": 10, "total_tokens": 20}})

    def respond(request, n):
        if outcome in _T4_REFUSED_BEFORE_SENDING:
            raise AssertionError(f"{outcome}: a request refused before sending reached "
                                 f"the network")
        if outcome == "success":
            return answer()
        if outcome == "HTTP 400":
            return httpx.Response(400, json={"error": {"message": "synthetic bad request",
                                                       "code": "invalid_request"}})
        if outcome == "HTTP 429":
            return httpx.Response(429, json={"error": {"message": "Rate limit reached",
                                                       "code": "rate_limit"}})
        if outcome == "timeout":
            raise httpx.ReadTimeout("synthetic timeout", request=request)
        if outcome == "500 then success":
            return (httpx.Response(500, json={"error": {"message": "synthetic upstream"}})
                    if n == 1 else answer())
        if outcome == "schema repaired":
            return answer('{"value":"bad"}') if n == 1 else answer()
        if outcome == "schema never satisfied":
            return answer('{"value":"bad"}')
        raise AssertionError(outcome)

    return respond


@pytest.mark.parametrize("outcome", _T4_OUTCOMES)
@pytest.mark.parametrize("route", ["replay", "shock", "free_text"])
def test_every_request_count_equals_what_reached_the_network(route, outcome, tmp_path,
                                                             monkeypatch):
    """
    THE DEFECT THIS PINS. Requests were counted at instructor's attempt hook and just before
    the raw SDK call - both before the transport's token guards - so a call the client
    refused itself was written as one transmitted request of unknown cost, in envelopes that
    passed the contract. Every figure a record or the session reports must now equal what
    actually reached the network: the envelope's count, its usage attempts, and the
    process's own total.
    """
    if route == "free_text" and outcome.startswith("schema"):
        pytest.skip("a free-text call has no schema to repair")
    if route != "free_text" and outcome == "a parameter the adapter refuses":
        pytest.skip("the stages never pass a parameter the adapter refuses")
    import warnings
    import httpx
    import openai
    from pydantic import BaseModel
    import courseapi
    import unsw_ai
    import scenarios as sc

    class DispatchProbe(BaseModel):
        value: int

    sent = []
    respond = _t4_responder(outcome, structured=route != "free_text")

    def network(request):
        sent.append(request)
        return respond(request, len(sent))

    built = []

    def build(settings, **kwargs):
        proxy = unsw_ai.ProxyTransport(settings, inner=httpx.MockTransport(network),
                                       token_limiter=kwargs.get("token_limiter"))
        built.append(openai.OpenAI(api_key="synthetic", base_url="https://synthetic.invalid",
                                   max_retries=0, http_client=httpx.Client(transport=proxy)))
        return built[-1]

    guard = unsw_ai.TokenLimiter(
        max_tokens_per_minute=1_000_000,
        max_request_tokens=1 if outcome == "per-request cap" else 1_000_000,
        max_tokens_per_day=1 if outcome == "daily budget" else 10_000_000)
    settings = unsw_ai.ProxySettings(proxy_url="https://synthetic.invalid",
                                     access_code="synthetic", student_id="9999999",
                                     fallback_models=())
    monkeypatch.setattr(unsw_ai, "build_openai_client", build)
    monkeypatch.setattr(unsw_ai.time, "sleep", lambda *a, **k: None)
    monkeypatch.setattr(config, "ledger_add", lambda *a, **k: None)
    monkeypatch.setattr(dr, "RAW_DIR", tmp_path)
    monkeypatch.setattr(sc, "RAW_DIR", tmp_path)
    recorded = usage = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        client = unsw_ai.UNSWInstructor(settings=settings, token_limiter=guard)
        monkeypatch.setattr(unsw_ai, "get_client", lambda *a, **k: client)
        monkeypatch.setattr(dr, "client_", lambda *a, **k: courseapi.CourseClient())
        monkeypatch.setattr(sc, "client_", lambda *a, **k: courseapi.CourseClient())
        try:
            if route == "replay":
                dr._cached_call("probe", "system", "user", schema=DispatchProbe)
            elif route == "shock":
                sc.llm_parsed("system", "user", DispatchProbe)
            else:
                extra = {"seed": 1} if outcome == "a parameter the adapter refuses" else {}
                reply = courseapi.CourseClient().chat.completions.create(
                    messages=[{"role": "user", "content": "synthetic"}],
                    temperature=config.SAMPLING_TEMPERATURE, **extra)
                recorded = reply.provenance.get("transport_requests") or 0
                usage = getattr(reply, "call_usage", None)
        except Exception as exc:                          # noqa: BLE001
            if route == "free_text":
                recorded = courseapi.transport_attempts(exc) or 0
                usage = courseapi.failed_attempt_usage(exc)
    for sdk in built:
        sdk.close()

    where = f"{route} / {outcome}"
    if route != "free_text":
        written = sorted(tmp_path.glob("*.json"))
        assert len(written) == 1, f"{where}: {len(written)} envelope(s) written"
        record = json.loads(written[0].read_text(encoding="utf-8"))
        assert courseapi.envelope_contradictions(record) == [], where
        recorded = record.get("transport_requests") or 0
        usage = record.get("usage")
    assert recorded == len(sent), (
        f"{where}: {len(sent)} request(s) reached the network, but the record says "
        f"{recorded}")
    if usage is None:
        assert len(sent) == 0, f"{where}: requests were sent, yet no usage was recorded"
    else:
        assert usage["attempts_counted"] + usage["attempts_unknown"] == len(sent), (
            f"{where}: usage accounts for "
            f"{usage['attempts_counted'] + usage['attempts_unknown']} request(s); "
            f"{len(sent)} were sent")
    assert client.usage.summary()["requests"] == len(sent), (
        f"{where}: the session total reports {client.usage.summary()['requests']} "
        f"request(s); {len(sent)} were sent")
    if outcome in _T4_REFUSED_BEFORE_SENDING:
        assert client.usage.report() == "No requests recorded."


# -------------------------------------------------------------------------------------------
# A FINISHED BENCHMARK STAYS FINISHED  (recheck-7 U1)
# -------------------------------------------------------------------------------------------
# Completion was reconstructed from readable close lines alone. Every fresh request below is a
# REAL cache miss through `_cached_call`, the course adapter and the wrapper's real
# ProxyTransport; only the network behind them is scripted, and every request it receives is
# counted. The earlier damage tests exercised cached replay, which never reaches the decision
# to send one.

def _holdout_request(monkeypatch, tmp_path, label):
    """One real cache miss on the holdout. Returns (the exposure event or the refusal, requests)."""
    import warnings
    import httpx
    import openai
    from pydantic import BaseModel
    import courseapi
    import unsw_ai

    class HoldoutProbe(BaseModel):
        value: int

    sent = []

    def network(request):
        sent.append(request)
        return httpx.Response(200, json={
            "id": "resp_synthetic", "object": "response", "created_at": 1,
            "model": config.MODEL, "status": "completed", "incomplete_details": None,
            "output": [{"type": "function_call", "id": "f", "call_id": "c",
                        "name": "HoldoutProbe", "arguments": '{"value":1}',
                        "status": "completed"}],
            "usage": {"input_tokens": 10, "output_tokens": 10, "total_tokens": 20}})

    built = []

    def build(settings, **kwargs):
        proxy = unsw_ai.ProxyTransport(settings, inner=httpx.MockTransport(network),
                                       token_limiter=kwargs.get("token_limiter"))
        built.append(openai.OpenAI(api_key="synthetic", base_url="https://synthetic.invalid",
                                   max_retries=0, http_client=httpx.Client(transport=proxy)))
        return built[-1]

    settings = unsw_ai.ProxySettings(proxy_url="https://synthetic.invalid",
                                     access_code="synthetic", student_id="9999999",
                                     fallback_models=())
    raw = tmp_path / f"raw-{len(list(tmp_path.glob('raw-*')))}"
    raw.mkdir()
    with monkeypatch.context() as m:
        m.setattr(unsw_ai, "build_openai_client", build)
        m.setattr(unsw_ai.time, "sleep", lambda *a, **k: None)
        m.setattr(config, "ledger_add", lambda *a, **k: None)
        m.setattr(dr, "RAW_DIR", raw)
        m.setattr(dr, "_CURRENT_SAMPLE", "holdout")
        m.setattr(dr, "_EXPOSURE_RECORDED", False)
        m.setattr(dr, "_EXPOSURE_EVENT", None)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            client = unsw_ai.UNSWInstructor(settings=settings)
            m.setattr(unsw_ai, "get_client", lambda *a, **k: client)
            m.setattr(dr, "client_", lambda *a, **k: courseapi.CourseClient())
            try:
                dr._cached_call("probe", "system", f"synthetic holdout question: {label}",
                                schema=HoldoutProbe)
                outcome = dr._EXPOSURE_EVENT
            except RuntimeError as exc:
                outcome = exc
    for sdk in built:
        sdk.close()
    return outcome, len(sent)


_COMPLETION_LOSS = ("malformed close line", "incomplete close object",
                    "truncated close line", "missing close line")


@pytest.mark.parametrize("loss", _COMPLETION_LOSS)
def test_a_finished_benchmark_is_not_resumed_when_its_close_record_is_lost(
        cached_holdout, monkeypatch, tmp_path, loss):
    """
    THE DEFECT THIS PINS (U1). With a finished benchmark's close line corrupted and the damage
    declared, a genuine cache miss reopened the exposure: one new request went to the held-out
    meetings as a "resumed" run, the count and the allowance stayed at one, and submission
    accepted the result. A resume now needs the freeze's own record that the benchmark did
    not finish; a further look is a separate, authorised exposure.
    """
    w = cached_holdout
    dr.freeze_selection()
    first, sent = _holdout_request(monkeypatch, tmp_path, "the one look")
    assert sent == 1 and not first["resumed"]
    origin = first["event_id"]
    dr.close_holdout_exposure(origin, "benchmark completed")
    assert dr._read_selection()["exposure_stamps"][0]["closed_at"], (
        "the freeze must record that the benchmark finished")

    lines = w.lines()
    close_at = next(i for i, line in enumerate(lines)
                    if json.loads(line).get("type") == "close")
    newline_at_end = True
    if loss == "malformed close line":
        lines[close_at] = "{synthetic damaged close"
    elif loss == "incomplete close object":
        event = json.loads(lines[close_at])
        event.pop("at")
        lines[close_at] = json.dumps(event, sort_keys=True)
    elif loss == "truncated close line":
        lines[close_at] = lines[close_at][: len(lines[close_at]) // 2]
        newline_at_end = close_at != len(lines) - 1
    else:
        del lines[close_at]
    w.write(lines, newline_at_end=newline_at_end)

    with pytest.raises(RuntimeError, match="damaged|record of it finishing"):
        dr.require_readable_log()
    _declare()
    dr.require_readable_log()

    refused, sent = _holdout_request(monkeypatch, tmp_path, "a look after the loss")
    assert isinstance(refused, RuntimeError) and "authorises" in str(refused), refused
    assert sent == 0, f"{loss}: {sent} request(s) reached the held-out meetings"
    assert dr.known_exposure_count() == 1
    assert dr._read_selection()["authorised_exposures"] == 1

    # CACHED RECOVERY IS UNAFFECTED, and asks nothing.
    recovered = w.benchmark()
    assert recovered.attrs["exposure_event_id"] == origin
    assert w.requests["n"] == 0
    w.submit(w.report(recovered))

    # ANOTHER LOOK IS A SEPARATE, AUTHORISED EXPOSURE - with its own id, count and history.
    dr.freeze_selection(revalidate=True, note="synthetic: a declared second look")
    second, sent = _holdout_request(monkeypatch, tmp_path, "the declared second look")
    assert sent == 1 and not second["resumed"]
    assert second["event_id"] != origin and second["sequence"] == 2
    assert dr.known_exposure_count() == 2
    dr.close_holdout_exposure(second["event_id"], "benchmark completed")
    latest = w.benchmark()
    assert latest.attrs["exposure_event_id"] == second["event_id"]
    w.submit(w.report(latest))


def test_an_interrupted_benchmark_is_resumed_as_the_same_exposure(cached_holdout, monkeypatch,
                                                                 tmp_path):
    """The control for U1: a benchmark that genuinely never finished resumes, uncounted."""
    w = cached_holdout
    dr.freeze_selection()
    first, sent = _holdout_request(monkeypatch, tmp_path, "the look, interrupted")
    assert sent == 1 and not first["resumed"]
    assert dr._read_selection()["exposure_stamps"][0]["closed_at"] is None, (
        "the freeze must record, when the exposure opens, that its benchmark has not finished")

    again, sent = _holdout_request(monkeypatch, tmp_path, "the same look, resumed")
    assert sent == 1 and again["resumed"] and again["event_id"] == first["event_id"]
    assert dr.known_exposure_count() == 1
    dr.close_holdout_exposure(first["event_id"], "benchmark completed")
    done = w.benchmark()
    assert done.attrs["exposure_event_id"] == first["event_id"]
    w.submit(w.report(done))
    assert dr.evidence_status() == "prospective"


def test_an_open_benchmark_is_not_resumed_past_damage_that_may_be_its_close(
        cached_holdout, monkeypatch, tmp_path):
    dr.freeze_selection()
    _, sent = _holdout_request(monkeypatch, tmp_path, "the look, interrupted")
    assert sent == 1
    with dr.EXPOSURE_LOG.open("ab") as fh:
        fh.write(b"{synthetic damage after the open record\n")
    _declare()
    refused, sent = _holdout_request(monkeypatch, tmp_path, "an attempted resume")
    assert isinstance(refused, RuntimeError), refused
    assert "declared after it began" in str(refused)
    assert sent == 0 and dr.known_exposure_count() == 1


# -------------------------------------------------------------------------------------------
# A PREVIOUS RELEASE'S WORKSPACE KEEPS ITS BENCHMARK ORIGIN  (recheck-7 U2)
# -------------------------------------------------------------------------------------------
# NOT AN INVENTED LEGACY SCHEMA. Each workspace below is the selection and the log that the
# PREVIOUS RELEASE's own functions wrote (starter commit a89afa3, `src/decision_replay.py`:
# freeze_selection, record_holdout_exposure, close_holdout_exposure and run_dev(lost_log=True),
# run on synthetic samples), captured byte for byte. The selection is written back with
# `json.dumps(selection, indent=1)` and each log line as stored - exactly what that release
# wrote. Its selections carry each exposure's full open event in `exposures` and no stamps.

_PREVIOUS_RELEASE_WORKSPACES = {
    'two exposures, one freeze': {
        'ids': {'first': '2c52d4966381', 'second': '93db603dc29b'},
        'selection': ('{"freeze_id": "f06434206160", "config_hash": "67e653ace5b9", "strategy": "strati'
                      'fied", "k": 9, "n_seeds": 3, "call_index_base": 20260818, "max_output_tokens": 8'
                      '000, "frozen_at": "2026-09-13T03:31:09.468256+00:00", "authorised_exposures": 2,'
                      ' "authorisation": "synthetic: a declared second look", "note": "synthetic: a dec'
                      'lared second look", "history": [], "revalidated": true, "prior_exposure_declared'
                      '": false, "exposure_number": 2, "exposure_log": "replay_exposures.jsonl", "expos'
                      'ures": [{"at": "2026-09-13T03:31:09.480255+00:00", "authorisation": "prospective'
                      ' selection freeze", "call_index_base": 20260818, "event_id": "2c52d4966381", "fr'
                      'eeze_id": "f06434206160", "k": 9, "n_seeds": 3, "reason": "synthetic first look"'
                      ', "selection_config_hash": "67e653ace5b9", "sequence": 1, "strategy": "stratifie'
                      'd", "type": "open"}, {"at": "2026-09-13T03:31:09.520865+00:00", "authorisation":'
                      ' "synthetic: a declared second look", "call_index_base": 20260818, "event_id": "'
                      '93db603dc29b", "freeze_id": "f06434206160", "k": 9, "n_seeds": 3, "reason": "syn'
                      'thetic second look", "selection_config_hash": "67e653ace5b9", "sequence": 2, "st'
                      'rategy": "stratified", "type": "open"}], "exposure_event_ids": ["2c52d4966381", '
                      '"93db603dc29b"], "refrozen_at": "2026-09-13T03:31:09.509798+00:00"}'),
        'log': [
            ('{"at": "2026-09-13T03:31:09.480255+00:00", "authorisation": "prospective selecti'
             'on freeze", "call_index_base": 20260818, "event_id": "2c52d4966381", "freeze_id"'
             ': "f06434206160", "k": 9, "n_seeds": 3, "reason": "synthetic first look", "selec'
             'tion_config_hash": "67e653ace5b9", "sequence": 1, "strategy": "stratified", "typ'
             'e": "open"}'),
            ('{"at": "2026-09-13T03:31:09.493294+00:00", "event_id": "2c52d4966381", "outcome"'
             ': "benchmark completed", "type": "close"}'),
            ('{"at": "2026-09-13T03:31:09.520865+00:00", "authorisation": "synthetic: a declar'
             'ed second look", "call_index_base": 20260818, "event_id": "93db603dc29b", "freez'
             'e_id": "f06434206160", "k": 9, "n_seeds": 3, "reason": "synthetic second look", '
             '"selection_config_hash": "67e653ace5b9", "sequence": 2, "strategy": "stratified"'
             ', "type": "open"}'),
            ('{"at": "2026-09-13T03:31:09.532873+00:00", "event_id": "93db603dc29b", "outcome"'
             ': "benchmark completed", "type": "close"}'),
        ],
    },
    'two exposures, changed freeze': {
        'ids': {'first': '4e2d75f61cb8', 'second': '442f98a89e2c'},
        'selection': ('{"freeze_id": "c285377e7e39", "config_hash": "8e55564ed1c6", "strategy": "strati'
                      'fied", "k": 9, "n_seeds": 3, "call_index_base": 20260918, "max_output_tokens": 8'
                      '000, "frozen_at": "2026-09-13T03:31:09.594108+00:00", "authorised_exposures": 2,'
                      ' "authorisation": "synthetic: a declared second look", "note": "synthetic: a dec'
                      'lared second look", "history": [{"freeze_id": "da51a5df934a", "config_hash": "67'
                      'e653ace5b9", "strategy": "stratified", "k": 9, "frozen_at": "2026-09-13T03:31:09'
                      '.549076+00:00", "authorised_exposures": 1, "exposures_at_supersession": 1}], "re'
                      'validated": true, "prior_exposure_declared": false, "exposure_number": 2, "expos'
                      'ure_log": "replay_exposures.jsonl", "exposures": [{"at": "2026-09-13T03:31:09.56'
                      '5076+00:00", "authorisation": "prospective selection freeze", "call_index_base":'
                      ' 20260818, "event_id": "4e2d75f61cb8", "freeze_id": "da51a5df934a", "k": 9, "n_s'
                      'eeds": 3, "reason": "synthetic first look", "selection_config_hash": "67e653ace5'
                      'b9", "sequence": 1, "strategy": "stratified", "type": "open"}, {"at": "2026-09-1'
                      '3T03:31:09.607624+00:00", "authorisation": "synthetic: a declared second look", '
                      '"call_index_base": 20260918, "event_id": "442f98a89e2c", "freeze_id": "c285377e7'
                      'e39", "k": 9, "n_seeds": 3, "reason": "synthetic look under a new selection", "s'
                      'election_config_hash": "8e55564ed1c6", "sequence": 2, "strategy": "stratified", '
                      '"type": "open"}], "exposure_event_ids": ["4e2d75f61cb8", "442f98a89e2c"]}'),
        'log': [
            ('{"at": "2026-09-13T03:31:09.565076+00:00", "authorisation": "prospective selecti'
             'on freeze", "call_index_base": 20260818, "event_id": "4e2d75f61cb8", "freeze_id"'
             ': "da51a5df934a", "k": 9, "n_seeds": 3, "reason": "synthetic first look", "selec'
             'tion_config_hash": "67e653ace5b9", "sequence": 1, "strategy": "stratified", "typ'
             'e": "open"}'),
            ('{"at": "2026-09-13T03:31:09.577082+00:00", "event_id": "4e2d75f61cb8", "outcome"'
             ': "benchmark completed", "type": "close"}'),
            ('{"at": "2026-09-13T03:31:09.607624+00:00", "authorisation": "synthetic: a declar'
             'ed second look", "call_index_base": 20260918, "event_id": "442f98a89e2c", "freez'
             'e_id": "c285377e7e39", "k": 9, "n_seeds": 3, "reason": "synthetic look under a n'
             'ew selection", "selection_config_hash": "8e55564ed1c6", "sequence": 2, "strategy'
             '": "stratified", "type": "open"}'),
            ('{"at": "2026-09-13T03:31:09.623207+00:00", "event_id": "442f98a89e2c", "outcome"'
             ': "benchmark completed", "type": "close"}'),
        ],
    },
    'latest exposure declared lost by that release': {
        'ids': {'first': 'a84c6b975b47', 'second': '9d52b478dfd8'},
        'selection': ('{"freeze_id": "ca192df948fe", "config_hash": "67e653ace5b9", "strategy": "strati'
                      'fied", "k": 9, "n_seeds": 3, "call_index_base": 20260818, "max_output_tokens": 8'
                      '000, "frozen_at": "2026-09-13T03:31:09.640735+00:00", "authorised_exposures": 2,'
                      ' "authorisation": "synthetic: a declared second look", "note": "synthetic: the l'
                      'og could not be restored from version control", "history": [], "revalidated": tr'
                      'ue, "prior_exposure_declared": false, "exposure_number": 2, "exposure_log": "rep'
                      'lay_exposures.jsonl", "exposures": [{"at": "2026-09-13T03:31:09.654733+00:00", "'
                      'authorisation": "prospective selection freeze", "call_index_base": 20260818, "ev'
                      'ent_id": "a84c6b975b47", "freeze_id": "ca192df948fe", "k": 9, "n_seeds": 3, "rea'
                      'son": "synthetic first look", "selection_config_hash": "67e653ace5b9", "sequence'
                      '": 1, "strategy": "stratified", "type": "open"}], "exposure_event_ids": ["a84c6b'
                      '975b47", "9d52b478dfd8"], "refrozen_at": "2026-09-13T03:31:09.720116+00:00", "lo'
                      'g_loss": {"declared_at": "2026-09-13T03:31:09.716611+00:00", "note": "synthetic:'
                      ' the log could not be restored from version control", "lost_event_ids": ["9d52b4'
                      '78dfd8"], "damaged_lines_at_declaration": []}}'),
        'log': [
            ('{"at": "2026-09-13T03:31:09.654733+00:00", "authorisation": "prospective selecti'
             'on freeze", "call_index_base": 20260818, "event_id": "a84c6b975b47", "freeze_id"'
             ': "ca192df948fe", "k": 9, "n_seeds": 3, "reason": "synthetic first look", "selec'
             'tion_config_hash": "67e653ace5b9", "sequence": 1, "strategy": "stratified", "typ'
             'e": "open"}'),
            ('{"at": "2026-09-13T03:31:09.667424+00:00", "event_id": "a84c6b975b47", "outcome"'
             ': "benchmark completed", "type": "close"}'),
            ('{"at": "2026-09-13T03:31:09.716611+00:00", "damaged_lines_at_declaration": [], "'
             'event_id": "7a8da7cf7a2c", "freeze_id": "ca192df948fe", "lost_event_ids": ["9d52'
             'b478dfd8"], "note": "synthetic: the log could not be restored from version contr'
             'ol", "selection_config_hash": "67e653ace5b9", "type": "log_loss_declared"}'),
        ],
    },
    'interrupted exposure': {
        'ids': {'first': 'cbb783523e64'},
        'selection': ('{"freeze_id": "0b007db4ad21", "config_hash": "67e653ace5b9", "strategy": "strati'
                      'fied", "k": 9, "n_seeds": 3, "call_index_base": 20260818, "max_output_tokens": 8'
                      '000, "frozen_at": "2026-09-13T03:31:09.739136+00:00", "authorised_exposures": 1,'
                      ' "authorisation": "prospective selection freeze", "note": "", "history": [], "re'
                      'validated": false, "prior_exposure_declared": false, "exposure_number": 1, "expo'
                      'sure_log": "replay_exposures.jsonl", "exposures": [{"at": "2026-09-13T03:31:09.7'
                      '51331+00:00", "authorisation": "prospective selection freeze", "call_index_base"'
                      ': 20260818, "event_id": "cbb783523e64", "freeze_id": "0b007db4ad21", "k": 9, "n_'
                      'seeds": 3, "reason": "synthetic look, interrupted", "selection_config_hash": "67'
                      'e653ace5b9", "sequence": 1, "strategy": "stratified", "type": "open"}], "exposur'
                      'e_event_ids": ["cbb783523e64"]}'),
        'log': [
            ('{"at": "2026-09-13T03:31:09.751331+00:00", "authorisation": "prospective selecti'
             'on freeze", "call_index_base": 20260818, "event_id": "cbb783523e64", "freeze_id"'
             ': "0b007db4ad21", "k": 9, "n_seeds": 3, "reason": "synthetic look, interrupted",'
             ' "selection_config_hash": "67e653ace5b9", "sequence": 1, "strategy": "stratified'
             '", "type": "open"}'),
        ],
    },
}


def _previous_release(name, monkeypatch):
    """Write a Replay workspace exactly as the previous release wrote it; return its ids."""
    fixture = _PREVIOUS_RELEASE_WORKSPACES[name]
    selection = json.loads(fixture["selection"])
    dr.SELECTION_STAMP.write_bytes(json.dumps(selection, indent=1).encode("utf-8"))
    dr.EXPOSURE_LOG.write_bytes("".join(line + "\n" for line in fixture["log"]).encode("utf-8"))
    monkeypatch.setattr(dr, "selection_config_hash", lambda: selection["config_hash"])
    return dict(fixture["ids"])


def _report_naming(event_id):
    return {"selection": dr._read_selection(),
            "exposure": {"events": dr.exposure_events(),
                         "evidence_status": dr.evidence_status(),
                         "evidence_flags": dr.evidence_flags(),
                         "holdout_event_id": event_id}}


_PREVIOUS_RELEASE_LOSSES = [
    ("two exposures, one freeze", "the later exposure"),
    ("two exposures, one freeze", "the earlier exposure"),
    ("two exposures, one freeze", "the whole log"),
    ("two exposures, changed freeze", "the later exposure"),
    ("two exposures, changed freeze", "the whole log"),
]


@pytest.mark.parametrize("saved_first", [False, True],
                         ids=["read as saved", "after this release saves it"])
@pytest.mark.parametrize("workspace,lost", _PREVIOUS_RELEASE_LOSSES)
def test_a_previous_release_workspace_keeps_its_benchmark_origin_through_a_loss(
        cached_holdout, monkeypatch, workspace, lost, saved_first):
    """
    THE DEFECT THIS PINS (U2). The origin was rebuilt from this release's stamps and the
    readable log, and the previous release wrote no stamps: its `exposures` list, which holds
    each exposure's freeze and sequence, was ignored - and then overwritten by the
    declaration. After a complete loss the cached benchmark named no exposure; after losing
    the later one it named the earlier, and submission accepted that.
    """
    w = cached_holdout
    ids = _previous_release(workspace, monkeypatch)
    reference = w.benchmark().to_dict("records")
    ids = _previous_release(workspace, monkeypatch)      # back to the bytes that release wrote
    if saved_first:
        dr.freeze_selection()                            # an ordinary --dev, nothing lost yet

    if lost == "the whole log":
        dr.EXPOSURE_LOG.unlink()
    else:
        gone = ids["second"] if lost == "the later exposure" else ids["first"]
        w.write(_without_event(w.lines(), gone))
    with pytest.raises(RuntimeError, match="no longer contains"):
        dr.require_readable_log()
    _declare()

    after = w.benchmark()
    assert after.attrs["exposure_event_id"] == ids["second"], (
        f"{workspace}, {lost}: the benchmark was bound to "
        f"{after.attrs['exposure_event_id']!r}, not the latest exposure {ids['second']!r}")
    assert after.to_dict("records") == reference, "recovery changed the benchmark's numbers"
    assert dr.known_exposure_count() == 2
    assert dr._read_selection()["authorised_exposures"] == 2
    assert w.requests["n"] == 0
    report = w.report(after)
    w.submit(report)
    older = json.loads(json.dumps(report))
    older["exposure"]["holdout_event_id"] = ids["first"]
    with pytest.raises(AssertionError):
        w.submit(older)


def test_an_origin_the_previous_release_left_undescribed_is_withheld_not_guessed(
        cached_holdout, monkeypatch):
    """
    GENUINELY ABSENT METADATA. The previous release declared the later exposure lost and
    rewrote its selection without that exposure's open event, so nothing records its freeze
    or order. The origin is withheld rather than settled on the older exposure that survives,
    and the supported way forward is a declared origin the flags report.
    """
    w = cached_holdout
    ids = _previous_release("latest exposure declared lost by that release", monkeypatch)
    dr.require_readable_log()
    origin, doubtful = dr.originating_exposure()
    assert origin is None and doubtful == [ids["second"]], (origin, doubtful)
    with pytest.raises(RuntimeError, match="cannot be bound"):
        w.benchmark()
    assert w.requests["n"] == 0
    for claimed in (ids["first"], ids["second"]):
        with pytest.raises(AssertionError, match="cannot be bound"):
            w.submit(_report_naming(claimed))

    with pytest.raises(RuntimeError, match="not an exposure that could"):
        dr.declare_origin("000000000000", "synthetic")
    with pytest.raises(RuntimeError, match="how you know"):
        dr.declare_origin(ids["second"], " ")
    dr.run_dev(origin=ids["second"],
               note="synthetic: the team's own record of that run names this exposure")
    after = w.benchmark()
    assert after.attrs["exposure_event_id"] == ids["second"]
    flags = dr.evidence_flags()
    assert flags["origin_declared"] is True and flags["status"] == "previously_exposed"
    assert dr.known_exposure_count() == 2
    assert dr._read_selection()["authorised_exposures"] == 2
    assert w.requests["n"] == 0
    report = w.report(after)
    w.submit(report)
    hidden = json.loads(json.dumps(report))
    hidden["exposure"]["evidence_flags"]["origin_declared"] = False
    with pytest.raises(AssertionError, match="contradicts the evidence"):
        w.submit(hidden)


def test_an_origin_the_records_establish_cannot_be_declared_over(cached_holdout):
    dr.freeze_selection()
    origin = cached_holdout.expose("the one look")
    with pytest.raises(RuntimeError, match="no origin to declare"):
        dr.declare_origin(origin, "synthetic: an attempt to restate a known origin")


@pytest.mark.parametrize("field,value", [("sequence", 5), ("freeze_id", "000000000000"),
                                         ("at", "2026-01-01T00:00:00+00:00"),
                                         ("selection_config_hash", "000000000000")])
def test_contradictory_saved_metadata_is_refused_not_resolved(cached_holdout, monkeypatch,
                                                              field, value):
    """A saved exposure record that disagrees with the log, with its own id or with its
    freeze's configuration is a contradiction - refused by every consumer, never settled."""
    w = cached_holdout
    ids = _previous_release("two exposures, one freeze", monkeypatch)
    selection = json.loads(dr.SELECTION_STAMP.read_text(encoding="utf-8"))
    selection["exposures"][-1][field] = value
    dr.SELECTION_STAMP.write_text(json.dumps(selection, indent=1), encoding="utf-8")
    with pytest.raises(RuntimeError, match="contradict"):
        dr.require_readable_log()

    w.write(_without_event(w.lines(), ids["second"]))
    with pytest.raises(RuntimeError, match="contradict"):
        dr.require_readable_log()
    with pytest.raises(RuntimeError, match="contradict"):
        _declare()
    with pytest.raises(AssertionError, match="contradict"):
        w.submit(_report_naming(ids["second"]))


def test_a_previous_release_exposure_left_open_is_not_resumed_on_a_missing_close(
        cached_holdout, monkeypatch, tmp_path):
    """
    That release kept no completion state, so a close line that is absent is all that says
    its benchmark did not finish - and absence is not evidence. Cached replay still finishes
    it without asking anything; a further request is a new, declared exposure.
    """
    w = cached_holdout
    ids = _previous_release("interrupted exposure", monkeypatch)
    refused, sent = _holdout_request(monkeypatch, tmp_path, "an attempted resume")
    assert isinstance(refused, RuntimeError), refused
    assert "kept no completion state" in str(refused)
    assert sent == 0 and dr.known_exposure_count() == 1

    done = w.benchmark()
    assert done.attrs["exposure_event_id"] == ids["first"] and w.requests["n"] == 0
    w.submit(w.report(done))
    dr.freeze_selection(revalidate=True, note="synthetic: a declared second look")
    second, sent = _holdout_request(monkeypatch, tmp_path, "the declared second look")
    assert sent == 1 and not second["resumed"] and second["sequence"] == 2


# -------------------------------------------------------------------------------------------
# ONE VALIDITY RULE FOR THE RUNTIME AND FOR SUBMISSION  (recheck-7 U3)
# -------------------------------------------------------------------------------------------

@pytest.mark.parametrize("deleted", ["an earlier exposure of this freeze",
                                     "an exposure of an earlier freeze",
                                     "the reported exposure"])
def test_submission_refuses_a_shortened_history_the_runtime_refuses(cached_holdout,
                                                                    monkeypatch, deleted):
    """
    THE DEFECT THIS PINS (U3). The submission check tested contradictions and undeclared
    damage, never undeclared MISSING exposures: with an earlier exposure's records deleted
    and the report regenerated from the shortened files, submission passed a history the
    Replay stage refused. Declaring exactly what was lost passes both, qualified.
    """
    w = cached_holdout
    dr.freeze_selection()
    first = w.expose("the first look")
    if deleted == "an exposure of an earlier freeze":
        monkeypatch.setattr(config, "CALL_INDEX_BASE", config.CALL_INDEX_BASE + 100)
    dr.freeze_selection(revalidate=True, note="synthetic: a declared second look")
    second = w.expose("the second look")
    w.write(_without_event(w.lines(), second if deleted == "the reported exposure" else first))

    with pytest.raises(RuntimeError, match="no longer contains"):
        dr.require_readable_log()
    with pytest.raises(AssertionError, match="no longer contains"):
        w.submit(_report_naming(second))

    _declare()
    dr.require_readable_log()
    after = w.benchmark()
    assert after.attrs["exposure_event_id"] == second
    w.submit(w.report(after))
    assert dr.evidence_status() == "previously_exposed"


def _history_step(w, step, saved):
    if step == "drop the earlier exposure":
        w.write(_without_event(w.lines(), saved["first"]))
    elif step == "drop the earlier close":
        w.write([line for line in w.lines()
                 if not (json.loads(line).get("type") == "close"
                         and json.loads(line).get("event_id") == saved["first"])])
    elif step == "delete the log":
        dr.EXPOSURE_LOG.unlink()
    elif step == "declare":
        _declare()
    elif step == "restore":
        w.write(saved["original"] + w.lines()[len(saved["kept"]):])
    elif step == "damage":
        with dr.EXPOSURE_LOG.open("ab") as fh:
            fh.write(b"{synthetic damage\n")
    elif step == "damage again":
        with dr.EXPOSURE_LOG.open("ab") as fh:
            fh.write(b"{synthetic damage added after the declaration\n")
    elif step == "contradict the earlier open":
        event = json.loads(saved["original"][0])
        event["reason"] = "a different account of the same exposure"
        w.write(w.lines() + [json.dumps(event, sort_keys=True)])
    elif step == "contradict the earlier stamp":
        selection = dr._read_selection()
        selection["exposure_stamps"][0]["sequence"] = 7
        dr.SELECTION_STAMP.write_text(json.dumps(selection, indent=1), encoding="utf-8")
    else:
        raise AssertionError(step)
    if step.startswith("drop"):
        saved["kept"] = w.lines()


_HISTORY_MATRIX = [
    ("intact", (), "accepted"),
    ("missing earlier exposure", ("drop the earlier exposure",), "refused"),
    ("missing earlier exposure, declared", ("drop the earlier exposure", "declare"),
     "accepted"),
    ("missing earlier exposure, declared, restored",
     ("drop the earlier exposure", "declare", "restore"), "accepted"),
    ("whole log deleted", ("delete the log",), "refused"),
    ("whole log deleted, declared", ("delete the log", "declare"), "accepted"),
    ("malformed line", ("damage",), "refused"),
    ("malformed line, declared", ("damage", "declare"), "accepted"),
    ("malformed line, declared, then more damage", ("damage", "declare", "damage again"),
     "refused"),
    ("missing earlier completion record", ("drop the earlier close",), "refused"),
    ("missing earlier completion record, declared", ("drop the earlier close", "declare"),
     "accepted"),
    ("two accounts of one exposure", ("contradict the earlier open",), "refused"),
    ("stamp contradicts the log", ("contradict the earlier stamp",), "refused"),
]


@pytest.mark.parametrize("case,steps,expected", _HISTORY_MATRIX,
                         ids=[case for case, _, _ in _HISTORY_MATRIX])
def test_the_runtime_and_submission_agree_on_every_history(cached_holdout, case, steps,
                                                           expected):
    """Every history is judged by one rule: the runtime refuses it exactly when submission
    does. Each report is regenerated from the altered files and names the latest exposure."""
    w = cached_holdout
    dr.freeze_selection()
    first = w.expose("the first look")
    dr.freeze_selection(revalidate=True, note="synthetic: a declared second look")
    second = w.expose("the second look")
    saved = {"first": first, "original": w.lines(), "kept": w.lines()}
    for step in steps:
        _history_step(w, step, saved)

    try:
        dr.require_readable_log()
        runtime = "accepted"
    except RuntimeError:
        runtime = "refused"
    try:
        w.submit(_report_naming(second))
        submission = "accepted"
    except AssertionError:
        submission = "refused"
    assert (runtime, submission) == (expected, expected), (
        f"{case}: the runtime {runtime} it and submission {submission} it; both should "
        f"have {expected} it")


def test_a_recorded_time_without_an_offset_is_compared_rather_than_crashing():
    """Every writer records UTC with an offset; a hand-edited time without one reads as UTC."""
    earlier = dr._when("2026-09-13T03:00:00")
    later = dr._when("2026-09-13T04:00:00+00:00")
    assert earlier is not None and earlier < later
    assert dr._when("not a time") is None and dr._when(None) is None


# -------------------------------------------------------------------------------------------
# A SUPERSEDED EXPOSURE IS NEVER RESUMED  (recheck-8 V1)
# -------------------------------------------------------------------------------------------
# Restoring an interrupted exposure's records after its declared replacement had finished made
# it resumable again: a real cache miss sent a new request under the old id, and a cached
# regeneration then named the replacement and passed submission. Every run below is the real
# `evaluate_all` or `_cached_call` through the course adapter and the wrapper's ProxyTransport.
# Requests are counted where they reach the scripted network, and the expected ids and counts
# are written out here rather than derived from the module under test.

def _run_benchmark(monkeypatch, tmp_path, question):
    """The real holdout benchmark, whose one call is a real `_cached_call`. The cache persists
    within a test, so a question already answered is replayed. Returns (frame or refusal,
    requests that reached the network)."""
    import warnings
    import httpx
    import openai
    from pydantic import BaseModel
    import courseapi
    import unsw_ai

    class HoldoutProbe(BaseModel):
        value: int

    sent = []

    def network(request):
        sent.append(request)
        return httpx.Response(200, json={
            "id": "resp_synthetic", "object": "response", "created_at": 1,
            "model": config.MODEL, "status": "completed", "incomplete_details": None,
            "output": [{"type": "function_call", "id": "f", "call_id": "c",
                        "name": "HoldoutProbe", "arguments": '{"value":1}',
                        "status": "completed"}],
            "usage": {"input_tokens": 10, "output_tokens": 10, "total_tokens": 20}})

    built = []

    def build(settings, **kwargs):
        proxy = unsw_ai.ProxyTransport(settings, inner=httpx.MockTransport(network),
                                       token_limiter=kwargs.get("token_limiter"))
        built.append(openai.OpenAI(api_key="synthetic", base_url="https://synthetic.invalid",
                                   max_retries=0, http_client=httpx.Client(transport=proxy)))
        return built[-1]

    def recommend(*a, **k):
        dr._cached_call("probe", "system", f"synthetic holdout question: {question}",
                        schema=HoldoutProbe)
        return dict(_CACHED_RECOMMENDATION)

    settings = unsw_ai.ProxySettings(proxy_url="https://synthetic.invalid",
                                     access_code="synthetic", student_id="9999999",
                                     fallback_models=())
    raw = tmp_path / "benchmark-raw"
    raw.mkdir(exist_ok=True)
    with monkeypatch.context() as m:
        m.setattr(unsw_ai, "build_openai_client", build)
        m.setattr(unsw_ai.time, "sleep", lambda *a, **k: None)
        m.setattr(config, "ledger_add", lambda *a, **k: None)
        m.setattr(dr, "RAW_DIR", raw)
        m.setattr(dr, "recommend", recommend)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            client = unsw_ai.UNSWInstructor(settings=settings)
            m.setattr(unsw_ai, "get_client", lambda *a, **k: client)
            m.setattr(dr, "client_", lambda *a, **k: courseapi.CourseClient())
            try:
                outcome = dr.evaluate_all(meetings=["2020-02-01"], strategies=["recent"],
                                          n_seeds=1, max_workers=1, sample="holdout")
            except RuntimeError as exc:
                outcome = exc
    for sdk in built:
        sdk.close()
    return outcome, len(sent)


_DAMAGED_LINE = b"{synthetic damaged line\n"


def _restore(saved, damage=None):
    """Put an earlier exposure's records back, keeping every later record exactly as it is."""
    now = dr.EXPOSURE_LOG.read_bytes() if dr.EXPOSURE_LOG.exists() else b""
    if damage and saved + damage in now:
        dr.EXPOSURE_LOG.write_bytes(now.replace(saved + damage, saved, 1))
    else:
        dr.EXPOSURE_LOG.write_bytes(saved + now)


@pytest.mark.parametrize("replacement", ["finished", "still open"])
@pytest.mark.parametrize("loss", ["missing log", "damaged line"])
def test_restoring_a_superseded_exposure_never_restores_permission_to_sample(
        cached_holdout, monkeypatch, tmp_path, loss, replacement):
    """
    THE DEFECT THIS PINS (V1). Exposure A was interrupted, its log lost or damaged and
    declared, and a replacement B authorised and run. Restoring A's records made A resumable
    even after B had finished: a fresh benchmark sent one request as "resumed" A with count and
    allowance at two, and regenerating from the cache that request had populated named B and
    passed submission. With B still open, B - the latest exposure - resumes as itself.
    """
    w = cached_holdout
    dr.freeze_selection()
    a, sent = _holdout_request(monkeypatch, tmp_path, "the first look, interrupted")
    assert sent == 1 and not a["resumed"]
    saved = dr.EXPOSURE_LOG.read_bytes()
    if loss == "missing log":
        dr.EXPOSURE_LOG.unlink()
    else:
        with dr.EXPOSURE_LOG.open("ab") as fh:
            fh.write(_DAMAGED_LINE)
    _declare()
    dr.freeze_selection(revalidate=True, note="synthetic: a declared replacement look")

    if replacement == "finished":
        frame, sent = _run_benchmark(monkeypatch, tmp_path, "the replacement's question")
        assert sent == 1 and frame.attrs["exposure_was_fresh"]
        b = frame.attrs["exposure_event_id"]
    else:
        opened, sent = _holdout_request(monkeypatch, tmp_path, "the replacement, interrupted")
        assert sent == 1 and not opened["resumed"]
        b = opened["event_id"]
    assert b != a["event_id"]
    assert dr._read_selection()["exposure_stamps"][0].get("superseded_by") == b, (
        "the freeze must record, when B begins, that B superseded A")

    _restore(saved, _DAMAGED_LINE if loss == "damaged line" else None)
    dr.require_readable_log()
    assert dr.known_exposure_count() == 2
    assert dr._read_selection()["authorised_exposures"] == 2

    frame, sent = _run_benchmark(monkeypatch, tmp_path, "a question asked after the restoration")
    if replacement == "still open":
        assert sent == 1 and frame.attrs["exposure_was_fresh"]
        assert frame.attrs["exposure_event_id"] == b
        assert dr.known_exposure_count() == 2
        w.submit(w.report(frame))
        return

    assert isinstance(frame, RuntimeError) and "superseded" in str(frame), frame
    assert sent == 0, f"{sent} request(s) went to the held-out meetings under exposure A"
    assert dr.known_exposure_count() == 2
    assert dr._read_selection()["authorised_exposures"] == 2
    assert len(list((tmp_path / "benchmark-raw").glob("*.json"))) == 1, (
        "the refused question must leave no answer in the cache")
    with pytest.raises(AssertionError, match="never closed|latest exposure"):
        w.submit(_report_naming(a["event_id"]))

    # CACHED REGENERATION asks nothing, names B, and holds only B's answer.
    frame, sent = _run_benchmark(monkeypatch, tmp_path, "the replacement's question")
    assert sent == 0 and not frame.attrs["exposure_was_fresh"]
    assert frame.attrs["exposure_event_id"] == b
    w.submit(w.report(frame))

    # ANOTHER LOOK IS A NEW, AUTHORISED EXPOSURE: C, the third.
    dr.freeze_selection(revalidate=True, note="synthetic: a declared third look")
    frame, sent = _run_benchmark(monkeypatch, tmp_path, "a question asked after the restoration")
    assert sent == 1 and frame.attrs["exposure_was_fresh"]
    c = frame.attrs["exposure_event_id"]
    assert c not in (a["event_id"], b)
    assert dr.known_exposure_count() == 3
    assert dr._read_selection()["authorised_exposures"] == 3
    w.submit(w.report(frame))


def test_a_superseded_exposure_cannot_be_recorded_as_finished(cached_holdout, monkeypatch,
                                                              tmp_path):
    dr.freeze_selection()
    a, _ = _holdout_request(monkeypatch, tmp_path, "the first look, interrupted")
    with dr.EXPOSURE_LOG.open("ab") as fh:
        fh.write(_DAMAGED_LINE)
    _declare()
    dr.freeze_selection(revalidate=True, note="synthetic: a declared replacement look")
    b, sent = _holdout_request(monkeypatch, tmp_path, "the replacement")
    assert sent == 1 and not b["resumed"]
    before = dr.EXPOSURE_LOG.read_bytes()
    with pytest.raises(RuntimeError, match="superseded"):
        dr.close_holdout_exposure(a["event_id"], "benchmark completed", fresh=True)
    assert dr.EXPOSURE_LOG.read_bytes() == before, "a refused close must write nothing"


# -------------------------------------------------------------------------------------------
# WORKSPACES WRITTEN BY THE RELEASE THAT HAD THE DEFECT  (recheck-8 V1)
# -------------------------------------------------------------------------------------------
# Written byte for byte by starter commit fb152f3's own functions: its freeze, exposure
# records and declaration, and its own `evaluate_all` with real cache misses over a scripted
# network. Restoring the saved start of the log is the one file operation, as a team would do.
# That release recorded no supersession, so its workspaces are judged by exposure order.

_DEFECT_RELEASE_WORKSPACES = {
    'superseded exposure restored after its replacement finished': {
        'ids': {'first': '35afc66cc167', 'second': 'c115b0fab8ea'},
        'selection': ('{"freeze_id": "b3f144f32ad5", "config_hash": "67e653ace5b9", "strategy": "strati'
                      'fied", "k": 9, "n_seeds": 3, "call_index_base": 20260818, "max_output_tokens": 8'
                      '000, "frozen_at": "2026-09-13T04:48:28.543591+00:00", "authorised_exposures": 2,'
                      ' "authorisation": "synthetic: a declared replacement look", "note": "synthetic: '
                      'a declared replacement look", "history": [], "revalidated": true, "prior_exposur'
                      'e_declared": false, "exposure_number": 2, "exposure_log": "replay_exposures.json'
                      'l", "exposures": [{"at": "2026-09-13T04:48:29.550116+00:00", "authorisation": "s'
                      'ynthetic: a declared replacement look", "call_index_base": 20260818, "event_id":'
                      ' "c115b0fab8ea", "freeze_id": "b3f144f32ad5", "k": 9, "n_seeds": 3, "reason": "f'
                      'resh probe call on the holdout sample", "selection_config_hash": "67e653ace5b9",'
                      ' "sequence": 2, "strategy": "stratified", "type": "open"}], "exposure_stamps": ['
                      '{"event_id": "35afc66cc167", "freeze_id": "b3f144f32ad5", "sequence": 1, "at": "'
                      '2026-09-13T04:48:28.575238+00:00", "selection_config_hash": "67e653ace5b9", "clo'
                      'sed_at": null}, {"event_id": "c115b0fab8ea", "freeze_id": "b3f144f32ad5", "seque'
                      'nce": 2, "at": "2026-09-13T04:48:29.550116+00:00", "selection_config_hash": "67e'
                      '653ace5b9", "closed_at": "2026-09-13T04:48:30.274940+00:00"}], "exposure_event_i'
                      'ds": ["35afc66cc167", "c115b0fab8ea"], "log_loss": {"declared_at": "2026-09-13T0'
                      '4:48:28.598315+00:00", "note": "synthetic: the log could not be restored from ve'
                      'rsion control", "declarations": [{"declared_at": "2026-09-13T04:48:28.598315+00:'
                      '00", "note": "synthetic: the log could not be restored from version control", "l'
                      'ost_event_ids": ["35afc66cc167"], "lost_close_ids": [], "damaged_lines": []}], "'
                      'lost_event_ids": ["35afc66cc167"], "lost_close_ids": [], "damaged_lines": [], "d'
                      'amaged_lines_at_declaration": []}, "refrozen_at": "2026-09-13T04:48:28.625694+00'
                      ':00"}'),
        'log': [
            ('{"at": "2026-09-13T04:48:28.575238+00:00", "authorisation": "prospective selecti'
             'on freeze", "call_index_base": 20260818, "event_id": "35afc66cc167", "freeze_id"'
             ': "b3f144f32ad5", "k": 9, "n_seeds": 3, "reason": "fresh probe call on the holdo'
             'ut sample", "selection_config_hash": "67e653ace5b9", "sequence": 1, "strategy": '
             '"stratified", "type": "open"}'),
            ('{"at": "2026-09-13T04:48:28.598315+00:00", "damaged_lines": [], "event_id": "25c'
             '909aabf74", "freeze_id": "b3f144f32ad5", "lost_close_ids": [], "lost_event_ids":'
             ' ["35afc66cc167"], "note": "synthetic: the log could not be restored from versio'
             'n control", "selection_config_hash": "67e653ace5b9", "type": "log_loss_declared"'
             '}'),
            ('{"at": "2026-09-13T04:48:29.550116+00:00", "authorisation": "synthetic: a declar'
             'ed replacement look", "call_index_base": 20260818, "event_id": "c115b0fab8ea", "'
             'freeze_id": "b3f144f32ad5", "k": 9, "n_seeds": 3, "reason": "fresh probe call on'
             ' the holdout sample", "selection_config_hash": "67e653ace5b9", "sequence": 2, "s'
             'trategy": "stratified", "type": "open"}'),
            ('{"at": "2026-09-13T04:48:30.274940+00:00", "event_id": "c115b0fab8ea", "outcome"'
             ': "holdout benchmark completed on 1 meetings", "type": "close"}'),
        ],
    },
    'superseded exposure resumed by that release, then regenerated from cache': {
        'ids': {'first': '35afc66cc167', 'second': 'c115b0fab8ea'},
        'selection': ('{"freeze_id": "b3f144f32ad5", "config_hash": "67e653ace5b9", "strategy": "strati'
                      'fied", "k": 9, "n_seeds": 3, "call_index_base": 20260818, "max_output_tokens": 8'
                      '000, "frozen_at": "2026-09-13T04:48:28.543591+00:00", "authorised_exposures": 2,'
                      ' "authorisation": "synthetic: a declared replacement look", "note": "synthetic: '
                      'a declared replacement look", "history": [], "revalidated": true, "prior_exposur'
                      'e_declared": false, "exposure_number": 2, "exposure_log": "replay_exposures.json'
                      'l", "exposures": [{"at": "2026-09-13T04:48:29.550116+00:00", "authorisation": "s'
                      'ynthetic: a declared replacement look", "call_index_base": 20260818, "event_id":'
                      ' "c115b0fab8ea", "freeze_id": "b3f144f32ad5", "k": 9, "n_seeds": 3, "reason": "f'
                      'resh probe call on the holdout sample", "selection_config_hash": "67e653ace5b9",'
                      ' "sequence": 2, "strategy": "stratified", "type": "open"}], "exposure_stamps": ['
                      '{"event_id": "35afc66cc167", "freeze_id": "b3f144f32ad5", "sequence": 1, "at": "'
                      '2026-09-13T04:48:28.575238+00:00", "selection_config_hash": "67e653ace5b9", "clo'
                      'sed_at": "2026-09-13T04:48:30.341468+00:00"}, {"event_id": "c115b0fab8ea", "free'
                      'ze_id": "b3f144f32ad5", "sequence": 2, "at": "2026-09-13T04:48:29.550116+00:00",'
                      ' "selection_config_hash": "67e653ace5b9", "closed_at": "2026-09-13T04:48:30.2749'
                      '40+00:00"}], "exposure_event_ids": ["35afc66cc167", "c115b0fab8ea"], "log_loss":'
                      ' {"declared_at": "2026-09-13T04:48:28.598315+00:00", "note": "synthetic: the log'
                      ' could not be restored from version control", "declarations": [{"declared_at": "'
                      '2026-09-13T04:48:28.598315+00:00", "note": "synthetic: the log could not be rest'
                      'ored from version control", "lost_event_ids": ["35afc66cc167"], "lost_close_ids"'
                      ': [], "damaged_lines": []}], "lost_event_ids": ["35afc66cc167"], "lost_close_ids'
                      '": [], "damaged_lines": [], "damaged_lines_at_declaration": []}, "refrozen_at": '
                      '"2026-09-13T04:48:28.625694+00:00"}'),
        'log': [
            ('{"at": "2026-09-13T04:48:28.575238+00:00", "authorisation": "prospective selecti'
             'on freeze", "call_index_base": 20260818, "event_id": "35afc66cc167", "freeze_id"'
             ': "b3f144f32ad5", "k": 9, "n_seeds": 3, "reason": "fresh probe call on the holdo'
             'ut sample", "selection_config_hash": "67e653ace5b9", "sequence": 1, "strategy": '
             '"stratified", "type": "open"}'),
            ('{"at": "2026-09-13T04:48:28.598315+00:00", "damaged_lines": [], "event_id": "25c'
             '909aabf74", "freeze_id": "b3f144f32ad5", "lost_close_ids": [], "lost_event_ids":'
             ' ["35afc66cc167"], "note": "synthetic: the log could not be restored from versio'
             'n control", "selection_config_hash": "67e653ace5b9", "type": "log_loss_declared"'
             '}'),
            ('{"at": "2026-09-13T04:48:29.550116+00:00", "authorisation": "synthetic: a declar'
             'ed replacement look", "call_index_base": 20260818, "event_id": "c115b0fab8ea", "'
             'freeze_id": "b3f144f32ad5", "k": 9, "n_seeds": 3, "reason": "fresh probe call on'
             ' the holdout sample", "selection_config_hash": "67e653ace5b9", "sequence": 2, "s'
             'trategy": "stratified", "type": "open"}'),
            ('{"at": "2026-09-13T04:48:30.274940+00:00", "event_id": "c115b0fab8ea", "outcome"'
             ': "holdout benchmark completed on 1 meetings", "type": "close"}'),
            ('{"at": "2026-09-13T04:48:30.341468+00:00", "event_id": "35afc66cc167", "outcome"'
             ': "holdout benchmark completed on 1 meetings", "type": "close"}'),
            ('{"at": "2026-09-13T04:48:30.378611+00:00", "event_id": "c115b0fab8ea", "outcome"'
             ': "holdout benchmark completed on 1 meetings (replayed from committed cache)", "'
             'type": "close"}'),
        ],
    },
}


def _defect_release(name, monkeypatch):
    """Write a Replay workspace exactly as the release with the defect wrote it."""
    fixture = _DEFECT_RELEASE_WORKSPACES[name]
    selection = json.loads(fixture["selection"])
    dr.SELECTION_STAMP.write_bytes(json.dumps(selection, indent=1).encode("utf-8"))
    dr.EXPOSURE_LOG.write_bytes("".join(line + "\n" for line in fixture["log"]).encode("utf-8"))
    monkeypatch.setattr(dr, "selection_config_hash", lambda: selection["config_hash"])
    return dict(fixture["ids"])


def test_a_workspace_restored_under_the_defective_release_cannot_resample_its_old_exposure(
        cached_holdout, monkeypatch, tmp_path):
    w = cached_holdout
    ids = _defect_release("superseded exposure restored after its replacement finished",
                          monkeypatch)
    dr.require_readable_log()
    refused, sent = _holdout_request(monkeypatch, tmp_path, "a question after the restoration")
    assert isinstance(refused, RuntimeError) and "superseded" in str(refused), refused
    assert sent == 0 and dr.known_exposure_count() == 2
    assert dr._read_selection()["authorised_exposures"] == 2
    replayed = w.benchmark()
    assert replayed.attrs["exposure_event_id"] == ids["second"] and w.requests["n"] == 0
    w.submit(w.report(replayed))
    dr.freeze_selection(revalidate=True, note="synthetic: a declared third look")
    third, sent = _holdout_request(monkeypatch, tmp_path, "the declared third look")
    assert sent == 1 and not third["resumed"] and third["sequence"] == 3
    assert third["event_id"] not in ids.values() and dr.known_exposure_count() == 3


def test_a_history_where_the_defective_release_resampled_a_superseded_exposure_is_refused(
        cached_holdout, monkeypatch):
    """
    The same workspace after that release's fresh benchmark resumed A - one request - and its
    cached regeneration named B. The history still records A completed by a run that sent
    requests after B began, so the runtime and submission refuse it, whichever exposure a
    report names: a cached regeneration can no longer hide which exposure asked.
    """
    w = cached_holdout
    ids = _defect_release(
        "superseded exposure resumed by that release, then regenerated from cache", monkeypatch)
    with pytest.raises(RuntimeError, match="superseded"):
        dr.require_readable_log()
    for claimed in (ids["first"], ids["second"]):
        with pytest.raises(AssertionError, match="superseded"):
            w.submit(_report_naming(claimed))
    # AND STAYS REFUSED after this release regenerates the benchmark from the cache again.
    regenerated = w.benchmark()
    assert regenerated.attrs["exposure_event_id"] == ids["second"] and w.requests["n"] == 0
    with pytest.raises(AssertionError, match="superseded"):
        w.submit(w.report(regenerated))
    with pytest.raises(RuntimeError, match="superseded"):
        dr.require_readable_log()


# -------------------------------------------------------------------------------------------
# RECOVERY LIFECYCLES WITH THEIR EXPECTED OUTCOMES WRITTEN OUT  (recheck-8 V1)
# -------------------------------------------------------------------------------------------

_LIFECYCLES = {
    "an interrupted benchmark resumes and finishes": (None, [
        "freeze", "open A", "resume A", "close A", "replay A", "submit A", "count 1/1"]),
    "log lost, replacement finished, old records restored": (None, [
        "freeze", "open A", "save", "lose log", "declare", "authorise", "open B", "close B",
        "restore", "count 2/2", "refused", "replay B", "submit B", "authorise", "open C",
        "close C", "replay C", "submit C", "count 3/3"]),
    "line damaged, replacement finished, damage removed": (None, [
        "freeze", "open A", "save", "damage", "declare", "authorise", "open B", "close B",
        "restore", "count 2/2", "refused", "replay B", "submit B", "authorise", "open C",
        "count 3/3"]),
    "log lost, old records restored while the replacement is open": (None, [
        "freeze", "open A", "save", "lose log", "declare", "authorise", "open B", "restore",
        "resume B", "close B", "replay B", "submit B", "count 2/2"]),
    "line damaged, damage removed while the replacement is open": (None, [
        "freeze", "open A", "save", "damage", "declare", "authorise", "open B", "restore",
        "resume B", "close B", "replay B", "submit B", "count 2/2"]),
    "previous release: an exposure left open, then replaced": ("interrupted exposure", [
        "refused", "replay A", "submit A", "authorise", "open B", "close B", "refused",
        "authorise", "open C", "count 3/3"]),
    "defective release: superseded exposure restored": (
        "superseded exposure restored after its replacement finished", [
            "refused", "replay B", "submit B", "authorise", "open C", "close C", "submit C",
            "count 3/3"]),
}


@pytest.mark.parametrize("case", list(_LIFECYCLES))
def test_recovery_lifecycles_never_restore_superseded_sampling_permission(
        cached_holdout, monkeypatch, tmp_path, case):
    """
    Each lifecycle lists its steps with the expected outcome of each written beside it - which
    exposure a request opens or resumes, when a request is refused, which exposure a replay
    names, how many exposures and authorisations there are - so a shared helper computing
    something wrong cannot make the producer and the validator agree on it. Requests are
    counted at the scripted network; ids are the ones the runs actually recorded.
    """
    _run_lifecycle(cached_holdout, monkeypatch, tmp_path, case, *_LIFECYCLES[case])


# -------------------------------------------------------------------------------------------
# THE DECLARATION MESSAGE COUNTS EXPOSURES  (recheck-8 V2)
# -------------------------------------------------------------------------------------------

_DECLARATION_MESSAGES = [
    ("a damaged line", 1, 1, "1 damaged line"),
    ("a completion record", 1, 1, "1 missing completion record"),
    ("the records of the earlier of two exposures", 2, 2,
     "the missing records of 1 exposure"),
    ("the whole log", 2, 2, "the missing records of 2 exposures"),
]


@pytest.mark.parametrize("lost,known,allowance,covers", _DECLARATION_MESSAGES,
                         ids=[case[0] for case in _DECLARATION_MESSAGES])
def test_the_declaration_message_counts_exposures_not_the_records_it_names(
        cached_holdout, capsys, lost, known, allowance, covers):
    """
    THE DEFECT THIS PINS (V2). The message printed how many exposure ids the declaration named
    as the number still counted: a damaged line or a missing close - every open record still
    readable - printed "0 exposure(s) stay counted" while one did, and a partial loss printed
    only the missing part. The expected numbers are written out, not taken from the module.
    """
    w = cached_holdout
    dr.freeze_selection()
    first = w.expose("the first look")
    if known == 2:
        dr.freeze_selection(revalidate=True, note="synthetic: a declared second look")
        w.expose("the second look")
    if lost == "a damaged line":
        with dr.EXPOSURE_LOG.open("ab") as fh:
            fh.write(_DAMAGED_LINE)
    elif lost == "a completion record":
        w.write([line for line in w.lines()
                 if not (json.loads(line).get("type") == "close"
                         and json.loads(line).get("event_id") == first)])
    elif lost == "the records of the earlier of two exposures":
        w.write(_without_event(w.lines(), first))
    else:
        dr.EXPOSURE_LOG.unlink()
    capsys.readouterr()
    _declare()
    out = capsys.readouterr().out
    remains = "exposure remains" if known == 1 else "exposures remain"
    assert f"{known} {remains} counted against an allowance of {allowance}" in out, out
    assert f"This declaration covers {covers}." in out, out
    assert "0 exposure" not in out, out
    assert dr.known_exposure_count() == known


def test_a_later_declaration_says_what_it_adds_and_what_all_of_them_cover(cached_holdout,
                                                                          capsys):
    w = cached_holdout
    dr.freeze_selection()
    w.expose("the one look")
    with dr.EXPOSURE_LOG.open("ab") as fh:
        fh.write(_DAMAGED_LINE)
    _declare()
    with dr.EXPOSURE_LOG.open("ab") as fh:
        fh.write(b"{synthetic damage added after the first declaration\n")
    capsys.readouterr()
    _declare("synthetic: more of the log was damaged")
    out = capsys.readouterr().out
    assert "1 exposure remains counted against an allowance of 1" in out, out
    assert "This declaration covers 1 damaged line." in out, out
    assert "All 2 declarations on this freeze cover 2 damaged lines." in out, out


# -------------------------------------------------------------------------------------------
# ONE INTERPRETER FOR EVERY RECOVERY LIFECYCLE  (recheck-9)
# -------------------------------------------------------------------------------------------
# A lifecycle is a list of steps, each carrying its expected outcome: which exposure a request
# opens or resumes, when a request is refused, which exposure a benchmark or a replay names,
# and how many exposures and authorisations there are. The expectations never come from the
# module under test, so a shared helper computing something wrong cannot make the producer and
# the validator agree on it. Requests are counted where they reach the scripted network.
#
#   freeze | authorise | reselect (changed configuration, declared) | declare
#   open X | resume X | refused                      - one real cache miss, in this process
#   restart open X | restart resume X | restart refused | restart replay X  - a FRESH process
#   produce X with Q | produce refused with Q        - the real benchmark, a real cache miss
#   reproduce X with Q                               - the real benchmark, served from cache
#   replay X | submit X | close X | count K/N
#   save | lose log | damage | drop X | restore [twice | after | open X]

def _run_lifecycle(w, monkeypatch, tmp_path, case, workspace, steps):
    loaded = {}
    if workspace in _PREVIOUS_RELEASE_WORKSPACES:
        loaded = _previous_release(workspace, monkeypatch)
    elif workspace in _DEFECT_RELEASE_WORKSPACES:
        loaded = _defect_release(workspace, monkeypatch)
    elif workspace:
        loaded = _released(workspace, monkeypatch)
    ids = {label: loaded[key] for key, label in (("first", "A"), ("second", "B"))
           if key in loaded}
    base = config.CALL_INDEX_BASE
    saved = {}

    def event_of(line):
        try:
            return json.loads(line).get("event_id")
        except json.JSONDecodeError:
            return None

    for number, step in enumerate(steps):
        where = f"{case}, step {number + 1} ({step})"
        head, _, question = step.partition(" with ")
        verb, _, arg = head.partition(" ")
        if verb == "freeze":
            dr.freeze_selection()
        elif verb in ("open", "resume"):
            event, sent = _holdout_request(monkeypatch, tmp_path, where)
            assert not isinstance(event, Exception), f"{where}: {event}"
            assert sent == 1, f"{where}: {sent} request(s)"
            if verb == "open":
                assert not event["resumed"] and event["event_id"] not in ids.values(), where
                ids[arg] = event["event_id"]
            else:
                assert event["resumed"] and event["event_id"] == ids[arg], where
        elif verb == "refused":
            event, sent = _holdout_request(monkeypatch, tmp_path, where)
            assert isinstance(event, RuntimeError), f"{where}: {event}"
            assert sent == 0, f"{where}: {sent} request(s) reached the held-out meetings"
        elif verb == "restart":
            action, _, label = arg.partition(" ")
            seen = _restart(tmp_path, "replay" if action == "replay" else "request", where,
                            config.CALL_INDEX_BASE - base)
            if action == "refused":
                assert seen["outcome"] == "refused" and seen["requests"] == 0, f"{where}: {seen}"
            elif action == "replay":
                assert seen["outcome"] == "replayed" and seen["requests"] == 0, f"{where}: {seen}"
                assert seen["event_id"] == ids[label], f"{where}: {seen}"
            elif action == "open":
                assert seen["outcome"] == "new" and seen["requests"] == 1, f"{where}: {seen}"
                assert seen["event_id"] not in ids.values(), f"{where}: {seen}"
                ids[label] = seen["event_id"]
            elif action == "resume":
                assert seen["outcome"] == "resumed" and seen["requests"] == 1, f"{where}: {seen}"
                assert seen["event_id"] == ids[label], f"{where}: {seen}"
            else:
                raise AssertionError(f"unknown restart step {step!r}")
        elif verb in ("produce", "reproduce"):
            frame, sent = _run_benchmark(monkeypatch, tmp_path, question)
            if arg == "refused":
                assert isinstance(frame, RuntimeError) and sent == 0, f"{where}: {frame} {sent}"
                continue
            assert not isinstance(frame, Exception), f"{where}: {frame}"
            named = frame.attrs["exposure_event_id"]
            if verb == "reproduce":
                assert sent == 0 and not frame.attrs["exposure_was_fresh"], f"{where}: {sent}"
                assert named == ids[arg], f"{where}: named {named}"
            else:
                assert sent == 1 and frame.attrs["exposure_was_fresh"], f"{where}: {sent}"
                if arg in ids:
                    assert named == ids[arg], f"{where}: produced under {named}"
                else:
                    assert named not in ids.values(), f"{where}: produced under {named}"
                    ids[arg] = named
        elif verb == "close":
            dr.close_holdout_exposure(ids[arg], "benchmark completed")
        elif verb == "save":
            saved["log"] = dr.EXPOSURE_LOG.read_bytes()
        elif verb == "lose":
            dr.EXPOSURE_LOG.unlink()
        elif verb == "damage":
            with dr.EXPOSURE_LOG.open("ab") as fh:
                fh.write(_DAMAGED_LINE)
        elif verb == "drop":
            w.write([line for line in w.lines() if event_of(line) != ids[arg]])
        elif verb == "restore":
            now = dr.EXPOSURE_LOG.read_bytes() if dr.EXPOSURE_LOG.exists() else b""
            if arg == "twice":
                dr.EXPOSURE_LOG.write_bytes(saved["log"] * 2 + now)
            elif arg == "after":
                dr.EXPOSURE_LOG.write_bytes(now + saved["log"])
            elif arg.startswith("open "):
                label = arg.split()[1]
                opens = [line for line in saved["log"].decode("utf-8").splitlines()
                         if json.loads(line).get("type") == "open"
                         and json.loads(line).get("event_id") == ids[label]]
                dr.EXPOSURE_LOG.write_bytes(("\n".join(opens) + "\n").encode("utf-8") + now)
            else:
                _restore(saved["log"], _DAMAGED_LINE)
        elif verb == "declare":
            _declare()
        elif verb == "authorise":
            dr.freeze_selection(revalidate=True, note=f"synthetic: {case}, a further look")
        elif verb == "reselect":
            monkeypatch.setattr(config, "CALL_INDEX_BASE", config.CALL_INDEX_BASE + 100)
            dr.freeze_selection(revalidate=True, note=f"synthetic: {case}, re-selected")
        elif verb == "replay":
            before = w.requests["n"]
            frame = w.benchmark()
            assert frame.attrs["exposure_event_id"] == ids[arg], where
            assert w.requests["n"] == before, where
        elif verb == "submit":
            w.submit(_report_naming(ids[arg]))
        elif verb == "count":
            known, allowance = (int(x) for x in arg.split("/"))
            assert len(ids) == known, f"{where}: the lifecycle has seen {len(ids)} exposure(s)"
            assert dr.known_exposure_count() == known, where
            assert dr._read_selection()["authorised_exposures"] == allowance, where
        else:
            raise AssertionError(f"unknown step {step!r}")


# -------------------------------------------------------------------------------------------
# A CRASH AT A PERSISTENCE BOUNDARY, THEN A RESTART IN A FRESH PROCESS  (recheck-9)
# -------------------------------------------------------------------------------------------
# An exposure and its completion are each written in two steps: the log line, fsynced, then
# the selection. A process can die between the two. Each case fails the write at that boundary
# and restarts in a fresh process against the same files - nothing held in memory survives -
# with the real course adapter and ProxyTransport over a scripted network.

_RESTART_STEP = r"""
import json, sys, warnings
from pathlib import Path

src, selection, log, raw, action, question, shift, config_hash = sys.argv[1:9]
sys.path.insert(0, src)
import config, courseapi, unsw_ai
import decision_replay as dr
import httpx, openai
from pydantic import BaseModel

dr.SELECTION_STAMP, dr.EXPOSURE_LOG, dr.RAW_DIR = Path(selection), Path(log), Path(raw)
dr.dev_sample = lambda *a, **k: ["2020-01-01"]
dr.holdout_sample = lambda *a, **k: ["2020-02-01"]
dr.selection_config_hash = lambda: config_hash
config.CALL_INDEX_BASE += int(shift)
config.ledger_add = lambda *a, **k: None
unsw_ai.time.sleep = lambda *a, **k: None


class HoldoutProbe(BaseModel):
    value: int


sent = []


def network(request):
    sent.append(request)
    return httpx.Response(200, json={
        "id": "resp_synthetic", "object": "response", "created_at": 1,
        "model": config.MODEL, "status": "completed", "incomplete_details": None,
        "output": [{"type": "function_call", "id": "f", "call_id": "c",
                    "name": "HoldoutProbe", "arguments": '{"value":1}',
                    "status": "completed"}],
        "usage": {"input_tokens": 10, "output_tokens": 10, "total_tokens": 20}})


def build(settings, **kwargs):
    proxy = unsw_ai.ProxyTransport(settings, inner=httpx.MockTransport(network),
                                   token_limiter=kwargs.get("token_limiter"))
    return openai.OpenAI(api_key="synthetic", base_url="https://synthetic.invalid",
                         max_retries=0, http_client=httpx.Client(transport=proxy))


unsw_ai.build_openai_client = build
settings = unsw_ai.ProxySettings(proxy_url="https://synthetic.invalid",
                                 access_code="synthetic", student_id="9999999",
                                 fallback_models=())
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    client = unsw_ai.UNSWInstructor(settings=settings)
unsw_ai.get_client = lambda *a, **k: client
dr.client_ = lambda *a, **k: courseapi.CourseClient()

if action == "request":
    dr._CURRENT_SAMPLE, dr._EXPOSURE_RECORDED, dr._EXPOSURE_EVENT = "holdout", False, None
    try:
        dr._cached_call("probe", "system", question, schema=HoldoutProbe)
        event = dr._EXPOSURE_EVENT
        result = {"outcome": "resumed" if event["resumed"] else "new",
                  "event_id": event["event_id"], "sequence": event["sequence"]}
    except RuntimeError as exc:
        result = {"outcome": "refused", "message": str(exc)}
else:
    dr.recommend = lambda *a, **k: {"ok": True, "result": {"recommendation": "hold",
                                                           "size_bp": 0, "confidence": 0.8},
                                    "shot_mix": {"hold_share": 1.0}}
    dr.feasible_everywhere = lambda meetings, strategies, k: (meetings, {})
    dr.actual_decision = lambda *a, **k: {"word": "hold", "size_bp": 0, "decision": 0}
    frame = dr.evaluate_all(meetings=["2020-02-01"], strategies=["recent"], n_seeds=1,
                            max_workers=1, sample="holdout")
    result = {"outcome": "replayed", "event_id": frame.attrs["exposure_event_id"]}
result.update(requests=len(sent), known=dr.known_exposure_count(),
              allowance=dr._read_selection().get("authorised_exposures"))
print("RESULT " + json.dumps(result))
"""


def _restart(tmp_path, action="request", question="a request after the restart", shift=0):
    """One step in a FRESH PROCESS against the same selection and log: what it saw - the
    outcome, the event, the requests that reached the network, the count and allowance."""
    import subprocess
    script = tmp_path / "restart_step.py"
    script.write_text(_RESTART_STEP, encoding="utf-8")
    raw = tmp_path / f"restart-raw-{len(list(tmp_path.glob('restart-raw-*')))}"
    raw.mkdir()
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "DOTENV_PATH": "none"}
    env.pop("OPENAI_API_KEY", None)
    src = pathlib.Path(__file__).resolve().parent.parent / "src"
    run = subprocess.run(
        [sys.executable, str(script), str(src), str(dr.SELECTION_STAMP), str(dr.EXPOSURE_LOG),
         str(raw), action, question, str(shift), dr.selection_config_hash()],
        capture_output=True, text=True, env=env, timeout=600)
    line = next((l for l in run.stdout.splitlines() if l.startswith("RESULT ")), None)
    assert line, (f"the restarted process reported nothing (exit {run.returncode}):\n"
                  f"{run.stdout[-1500:]}\n{run.stderr[-3000:]}")
    return json.loads(line[len("RESULT "):])


class _Crash(RuntimeError):
    """The process dying at a persistence boundary."""


def _crash_selection_writes(monkeypatch):
    """Make selection writes fail as a crash would; returns the function that stops it."""
    real = config.atomic_write_text

    def write(path, text, *args, **kwargs):
        if pathlib.Path(path) == pathlib.Path(dr.SELECTION_STAMP):
            raise _Crash("synthetic crash before the selection was updated")
        return real(path, text, *args, **kwargs)

    monkeypatch.setattr(config, "atomic_write_text", write)
    return lambda: monkeypatch.setattr(config, "atomic_write_text", real)


def test_a_crash_after_an_exposure_is_logged_counts_it_and_sends_nothing_on_restart(
        cached_holdout, monkeypatch, tmp_path):
    """The open record is durable and the selection never learns of it. The exposure still
    counts; with nothing recording that its benchmark had not finished, a restart sends
    nothing at the spent allowance, and a declared further look is a new exposure."""
    w = cached_holdout
    dr.freeze_selection()
    stop = _crash_selection_writes(monkeypatch)
    crashed, sent = _holdout_request(monkeypatch, tmp_path, "the first look")
    stop()
    assert isinstance(crashed, _Crash) and sent == 0, (crashed, sent)
    opened = [json.loads(line) for line in w.lines() if json.loads(line).get("type") == "open"]
    assert len(opened) == 1 and not dr._read_selection().get("exposure_stamps"), (
        "the crash must leave the exposure logged and unrecorded in the selection")

    after = _restart(tmp_path)
    assert after["outcome"] == "refused" and after["requests"] == 0, after
    assert after["known"] == 1 and after["allowance"] == 1, after

    dr.freeze_selection(revalidate=True, note="synthetic: a declared look after the crash")
    again = _restart(tmp_path)
    assert again["outcome"] == "new" and again["requests"] == 1, again
    assert again["sequence"] == 2 and again["event_id"] != opened[0]["event_id"], again
    assert again["known"] == 2 and again["allowance"] == 2, again


def test_a_crash_before_the_close_is_logged_leaves_a_missing_record_until_declared(
        cached_holdout, monkeypatch, tmp_path):
    """The freeze records a completion BEFORE the log does (recheck-10 W1), so the crash window
    is now the other way round: the selection says finished, the log has no close. That is a
    missing record - refused on restart until declared - never a benchmark that looks unfinished;
    once declared, a restart sends nothing and a restarted cached replay writes the close."""
    w = cached_holdout
    dr.freeze_selection()
    first, sent = _holdout_request(monkeypatch, tmp_path, "the one look")
    assert sent == 1 and not first["resumed"]
    real_append = dr._append_event

    def append(event):
        if event.get("type") == "close":
            raise _Crash("synthetic crash before the close was logged")
        return real_append(event)

    monkeypatch.setattr(dr, "_append_event", append)
    with pytest.raises(_Crash):
        dr.close_holdout_exposure(first["event_id"], "holdout benchmark completed on 1 meetings",
                                  fresh=True)
    monkeypatch.setattr(dr, "_append_event", real_append)
    assert dr._read_selection()["exposure_stamps"][0]["closed_at"], (
        "the selection must record the completion before the log line is written")
    assert not any(json.loads(line).get("type") == "close" for line in w.lines())

    after = _restart(tmp_path)
    assert after["outcome"] == "refused" and after["requests"] == 0, after
    assert "record of it finishing" in after["message"], after
    _declare()
    again = _restart(tmp_path)
    assert again["outcome"] == "refused" and again["requests"] == 0 and again["known"] == 1, again
    replayed = _restart(tmp_path, "replay")
    assert replayed["outcome"] == "replayed" and replayed["requests"] == 0, replayed
    assert replayed["event_id"] == first["event_id"], replayed
    assert any(json.loads(line).get("type") == "close" for line in w.lines())
    w.submit(_report_naming(first["event_id"]))


def test_a_workspace_left_by_the_log_first_close_order_keeps_the_benchmark_finished(
        cached_holdout, monkeypatch, tmp_path):
    """Files the released b825b4e writer left after a crash between its close line and its
    selection update: the readable close wins, a restart sends nothing, and a restarted cached
    replay records the completion in the selection."""
    w = cached_holdout
    ids = _released("b825b4e: completion logged, selection update lost in a crash", monkeypatch)
    assert dr._read_selection()["exposure_stamps"][0]["closed_at"] is None
    assert any(json.loads(line).get("type") == "close" for line in w.lines())

    after = _restart(tmp_path)
    assert after["outcome"] == "refused" and after["requests"] == 0 and after["known"] == 1, after
    replayed = _restart(tmp_path, "replay")
    assert replayed["outcome"] == "replayed" and replayed["requests"] == 0, replayed
    assert replayed["event_id"] == ids["first"], replayed
    assert dr._read_selection()["exposure_stamps"][0]["closed_at"], (
        "the restarted run must record the completion in the selection")
    again = _restart(tmp_path)
    assert again["outcome"] == "refused" and again["requests"] == 0, again
    w.submit(_report_naming(ids["first"]))


def test_a_crash_while_a_replacement_is_recorded_never_revives_what_it_superseded(
        cached_holdout, monkeypatch, tmp_path):
    """The replacement's open record is durable; its stamp and the supersession record written
    with it are lost. With the damage that blocked the old exposure gone as well, order alone
    says it was superseded - and a restart still sends nothing at the spent allowance."""
    w = cached_holdout
    dr.freeze_selection()
    first, sent = _holdout_request(monkeypatch, tmp_path, "the first look, interrupted")
    assert sent == 1
    with dr.EXPOSURE_LOG.open("ab") as fh:
        fh.write(_DAMAGED_LINE)
    _declare()
    dr.freeze_selection(revalidate=True, note="synthetic: a declared replacement look")
    stop = _crash_selection_writes(monkeypatch)
    crashed, sent = _holdout_request(monkeypatch, tmp_path, "the replacement")
    stop()
    assert isinstance(crashed, _Crash) and sent == 0, (crashed, sent)
    stamps = dr._read_selection()["exposure_stamps"]
    assert len(stamps) == 1 and not stamps[0].get("superseded_by"), stamps
    w.write([line for line in w.lines() if not line.startswith("{synthetic damaged line")])

    after = _restart(tmp_path)
    assert after["outcome"] == "refused" and after["requests"] == 0, after
    assert after["known"] == 2 and after["allowance"] == 2, after
    assert dr.open_exposure(dr._read_selection()["freeze_id"]) is None

    dr.freeze_selection(revalidate=True, note="synthetic: a declared third look")
    third = _restart(tmp_path)
    assert third["outcome"] == "new" and third["requests"] == 1 and third["sequence"] == 3, third
    assert third["event_id"] != first["event_id"], third


def test_an_interrupted_latest_exposure_resumes_after_a_restart(cached_holdout, monkeypatch,
                                                                tmp_path):
    """The control: every record written, the run interrupted - a fresh process resumes it."""
    dr.freeze_selection()
    first, sent = _holdout_request(monkeypatch, tmp_path, "the look, interrupted")
    assert sent == 1
    resumed = _restart(tmp_path)
    assert resumed["outcome"] == "resumed" and resumed["requests"] == 1, resumed
    assert resumed["event_id"] == first["event_id"] and resumed["known"] == 1, resumed


# -------------------------------------------------------------------------------------------
# BROADER RESTORATION SEQUENCES  (recheck-9)
# -------------------------------------------------------------------------------------------
# Duplicated and reordered restorations, a partial restoration, several replacements, and a
# replacement's own records lost after an earlier exposure was restored - with and without a
# changed configuration, and across a restart. Duplicates never raise the count, restoration
# never lowers it or revives sampling permission, and a fresh request uses only the latest
# authorised exposure. Producer output, cached replay and submission run together.

_RESTORATION_LIFECYCLES = {
    "original records restored twice": [
        "freeze", "open A", "save", "lose log", "declare", "authorise",
        "produce B with the replacement's question", "restore twice", "count 2/2", "refused",
        "produce refused with a question after the restoration",
        "reproduce B with the replacement's question", "submit B", "restart refused",
        "authorise", "produce C with a question after the restoration", "submit C",
        "count 3/3"],
    "original records restored after the later ones": [
        "freeze", "open A", "save", "lose log", "declare", "authorise",
        "produce B with the replacement's question", "restore after", "count 2/2", "refused",
        "reproduce B with the replacement's question", "submit B", "count 2/2"],
    "a finished exposure's open record restored alone": [
        "freeze", "produce A with the first question", "save", "lose log", "declare",
        "authorise", "produce B with the replacement's question", "restore open A",
        "count 2/2", "refused", "reproduce B with the replacement's question", "submit B",
        "count 2/2"],
    "three exposures, the earliest records restored": [
        "freeze", "open A", "save", "lose log", "declare", "authorise", "open B", "damage",
        "declare", "authorise", "produce C with the third question", "restore", "count 3/3",
        "refused", "restart refused", "reproduce C with the third question", "submit C",
        "authorise", "produce D with a fourth question", "submit D", "count 4/4"],
    "the replacement's records lost after the restoration": [
        "freeze", "open A", "save", "lose log", "declare", "authorise",
        "produce B with the replacement's question", "restore", "drop B", "declare",
        "count 2/2", "refused", "restart refused", "authorise", "restart open C", "replay C",
        "submit C", "count 3/3"],
    "the replacement's records lost after the restoration, under a changed freeze": [
        "freeze", "open A", "save", "lose log", "declare", "reselect",
        "produce B with the replacement's question", "restore", "drop B", "declare",
        "count 2/2", "refused", "restart refused", "authorise", "restart open C", "replay C",
        "submit C", "count 3/3"],
    "the latest of three exposures resumes after the restoration": [
        "freeze", "open A", "save", "lose log", "declare", "authorise", "open B", "damage",
        "declare", "authorise", "open C", "restore", "resume C", "restart resume C",
        "close C", "replay C", "submit C", "count 3/3"],
}


@pytest.mark.parametrize("case", list(_RESTORATION_LIFECYCLES))
def test_restoration_never_raises_counts_or_revives_superseded_sampling(cached_holdout,
                                                                        monkeypatch, tmp_path,
                                                                        case):
    _run_lifecycle(cached_holdout, monkeypatch, tmp_path, case, None,
                   _RESTORATION_LIFECYCLES[case])


# -------------------------------------------------------------------------------------------
# WORKSPACES WRITTEN BY EACH RELEASED VERSION  (recheck-9)
# -------------------------------------------------------------------------------------------
# Written byte for byte by the released starter commits e91cdbc, fb152f3 and 62eb7ed, running
# their own functions: freeze, exposure record, `_cached_call` through the course adapter and
# the wrapper's real ProxyTransport over a scripted network, and `evaluate_all`. For each: a
# benchmark completed by a fresh run, one whose request was made before an interruption and
# which a cached re-run completed, and one interrupted after its exposure was recorded. A
# migration regression shows only when an earlier release's files meet this reader.

_RELEASED_WORKSPACES = {
    'e91cdbc: completed': {
        'ids': {'first': 'b8e1d663363e'},
        'selection': ('{"freeze_id": "3c81ec3f6bdd", "config_hash": "67e653ace5b9", "strategy": "strati'
                      'fied", "k": 9, "n_seeds": 3, "call_index_base": 20260818, "max_output_tokens": 8'
                      '000, "frozen_at": "2026-09-13T06:11:52.525619+00:00", "authorised_exposures": 1,'
                      ' "authorisation": "prospective selection freeze", "note": "", "history": [], "re'
                      'validated": false, "prior_exposure_declared": false, "exposure_number": 1, "expo'
                      'sure_log": "replay_exposures.jsonl", "exposures": [{"at": "2026-09-13T06:11:53.4'
                      '68523+00:00", "authorisation": "prospective selection freeze", "call_index_base"'
                      ': 20260818, "event_id": "b8e1d663363e", "freeze_id": "3c81ec3f6bdd", "k": 9, "n_'
                      'seeds": 3, "reason": "fresh probe call on the holdout sample", "selection_config'
                      '_hash": "67e653ace5b9", "sequence": 1, "strategy": "stratified", "type": "open"}'
                      '], "exposure_stamps": [{"event_id": "b8e1d663363e", "freeze_id": "3c81ec3f6bdd",'
                      ' "sequence": 1, "at": "2026-09-13T06:11:53.468523+00:00", "selection_config_hash'
                      '": "67e653ace5b9"}], "exposure_event_ids": ["b8e1d663363e"]}'),
        'log': [
            ('{"at": "2026-09-13T06:11:53.468523+00:00", "authorisation": "prospective selecti'
             'on freeze", "call_index_base": 20260818, "event_id": "b8e1d663363e", "freeze_id"'
             ': "3c81ec3f6bdd", "k": 9, "n_seeds": 3, "reason": "fresh probe call on the holdo'
             'ut sample", "selection_config_hash": "67e653ace5b9", "sequence": 1, "strategy": '
             '"stratified", "type": "open"}'),
            ('{"at": "2026-09-13T06:11:54.265970+00:00", "event_id": "b8e1d663363e", "outcome"'
             ': "holdout benchmark completed on 1 meetings", "type": "close"}'),
        ],
    },
    'e91cdbc: completed from cache': {
        'ids': {'first': 'e89b6923d347'},
        'selection': ('{"freeze_id": "a820bba5a0f4", "config_hash": "67e653ace5b9", "strategy": "strati'
                      'fied", "k": 9, "n_seeds": 3, "call_index_base": 20260818, "max_output_tokens": 8'
                      '000, "frozen_at": "2026-09-13T06:11:54.291093+00:00", "authorised_exposures": 1,'
                      ' "authorisation": "prospective selection freeze", "note": "", "history": [], "re'
                      'validated": false, "prior_exposure_declared": false, "exposure_number": 1, "expo'
                      'sure_log": "replay_exposures.jsonl", "exposures": [{"at": "2026-09-13T06:11:54.3'
                      '06598+00:00", "authorisation": "prospective selection freeze", "call_index_base"'
                      ': 20260818, "event_id": "e89b6923d347", "freeze_id": "a820bba5a0f4", "k": 9, "n_'
                      'seeds": 3, "reason": "fresh probe call on the holdout sample", "selection_config'
                      '_hash": "67e653ace5b9", "sequence": 1, "strategy": "stratified", "type": "open"}'
                      '], "exposure_stamps": [{"event_id": "e89b6923d347", "freeze_id": "a820bba5a0f4",'
                      ' "sequence": 1, "at": "2026-09-13T06:11:54.306598+00:00", "selection_config_hash'
                      '": "67e653ace5b9"}], "exposure_event_ids": ["e89b6923d347"]}'),
        'log': [
            ('{"at": "2026-09-13T06:11:54.306598+00:00", "authorisation": "prospective selecti'
             'on freeze", "call_index_base": 20260818, "event_id": "e89b6923d347", "freeze_id"'
             ': "a820bba5a0f4", "k": 9, "n_seeds": 3, "reason": "fresh probe call on the holdo'
             'ut sample", "selection_config_hash": "67e653ace5b9", "sequence": 1, "strategy": '
             '"stratified", "type": "open"}'),
            ('{"at": "2026-09-13T06:11:54.352881+00:00", "event_id": "e89b6923d347", "outcome"'
             ': "holdout benchmark completed on 1 meetings (replayed from committed cache)", "'
             'type": "close"}'),
        ],
    },
    'e91cdbc: interrupted': {
        'ids': {'first': 'bc638c777f7f'},
        'selection': ('{"freeze_id": "6b3bc9ae724f", "config_hash": "67e653ace5b9", "strategy": "strati'
                      'fied", "k": 9, "n_seeds": 3, "call_index_base": 20260818, "max_output_tokens": 8'
                      '000, "frozen_at": "2026-09-13T06:11:54.368881+00:00", "authorised_exposures": 1,'
                      ' "authorisation": "prospective selection freeze", "note": "", "history": [], "re'
                      'validated": false, "prior_exposure_declared": false, "exposure_number": 1, "expo'
                      'sure_log": "replay_exposures.jsonl", "exposures": [{"at": "2026-09-13T06:11:54.3'
                      '82884+00:00", "authorisation": "prospective selection freeze", "call_index_base"'
                      ': 20260818, "event_id": "bc638c777f7f", "freeze_id": "6b3bc9ae724f", "k": 9, "n_'
                      'seeds": 3, "reason": "fresh probe call on the holdout sample", "selection_config'
                      '_hash": "67e653ace5b9", "sequence": 1, "strategy": "stratified", "type": "open"}'
                      '], "exposure_stamps": [{"event_id": "bc638c777f7f", "freeze_id": "6b3bc9ae724f",'
                      ' "sequence": 1, "at": "2026-09-13T06:11:54.382884+00:00", "selection_config_hash'
                      '": "67e653ace5b9"}], "exposure_event_ids": ["bc638c777f7f"]}'),
        'log': [
            ('{"at": "2026-09-13T06:11:54.382884+00:00", "authorisation": "prospective selecti'
             'on freeze", "call_index_base": 20260818, "event_id": "bc638c777f7f", "freeze_id"'
             ': "6b3bc9ae724f", "k": 9, "n_seeds": 3, "reason": "fresh probe call on the holdo'
             'ut sample", "selection_config_hash": "67e653ace5b9", "sequence": 1, "strategy": '
             '"stratified", "type": "open"}'),
        ],
    },
    'fb152f3: completed': {
        'ids': {'first': '56cdc95edaf1'},
        'selection': ('{"freeze_id": "e4366a3e0849", "config_hash": "67e653ace5b9", "strategy": "strati'
                      'fied", "k": 9, "n_seeds": 3, "call_index_base": 20260818, "max_output_tokens": 8'
                      '000, "frozen_at": "2026-09-13T06:11:54.417439+00:00", "authorised_exposures": 1,'
                      ' "authorisation": "prospective selection freeze", "note": "", "history": [], "re'
                      'validated": false, "prior_exposure_declared": false, "exposure_number": 1, "expo'
                      'sure_log": "replay_exposures.jsonl", "exposures": [{"at": "2026-09-13T06:11:54.4'
                      '33452+00:00", "authorisation": "prospective selection freeze", "call_index_base"'
                      ': 20260818, "event_id": "56cdc95edaf1", "freeze_id": "e4366a3e0849", "k": 9, "n_'
                      'seeds": 3, "reason": "fresh probe call on the holdout sample", "selection_config'
                      '_hash": "67e653ace5b9", "sequence": 1, "strategy": "stratified", "type": "open"}'
                      '], "exposure_stamps": [{"event_id": "56cdc95edaf1", "freeze_id": "e4366a3e0849",'
                      ' "sequence": 1, "at": "2026-09-13T06:11:54.433452+00:00", "selection_config_hash'
                      '": "67e653ace5b9", "closed_at": "2026-09-13T06:11:54.464321+00:00"}], "exposure_'
                      'event_ids": ["56cdc95edaf1"]}'),
        'log': [
            ('{"at": "2026-09-13T06:11:54.433452+00:00", "authorisation": "prospective selecti'
             'on freeze", "call_index_base": 20260818, "event_id": "56cdc95edaf1", "freeze_id"'
             ': "e4366a3e0849", "k": 9, "n_seeds": 3, "reason": "fresh probe call on the holdo'
             'ut sample", "selection_config_hash": "67e653ace5b9", "sequence": 1, "strategy": '
             '"stratified", "type": "open"}'),
            ('{"at": "2026-09-13T06:11:54.464321+00:00", "event_id": "56cdc95edaf1", "outcome"'
             ': "holdout benchmark completed on 1 meetings", "type": "close"}'),
        ],
    },
    'fb152f3: completed from cache': {
        'ids': {'first': 'b88fe6fe13a9'},
        'selection': ('{"freeze_id": "8c02a6246bd0", "config_hash": "67e653ace5b9", "strategy": "strati'
                      'fied", "k": 9, "n_seeds": 3, "call_index_base": 20260818, "max_output_tokens": 8'
                      '000, "frozen_at": "2026-09-13T06:11:54.492825+00:00", "authorised_exposures": 1,'
                      ' "authorisation": "prospective selection freeze", "note": "", "history": [], "re'
                      'validated": false, "prior_exposure_declared": false, "exposure_number": 1, "expo'
                      'sure_log": "replay_exposures.jsonl", "exposures": [{"at": "2026-09-13T06:11:54.5'
                      '10338+00:00", "authorisation": "prospective selection freeze", "call_index_base"'
                      ': 20260818, "event_id": "b88fe6fe13a9", "freeze_id": "8c02a6246bd0", "k": 9, "n_'
                      'seeds": 3, "reason": "fresh probe call on the holdout sample", "selection_config'
                      '_hash": "67e653ace5b9", "sequence": 1, "strategy": "stratified", "type": "open"}'
                      '], "exposure_stamps": [{"event_id": "b88fe6fe13a9", "freeze_id": "8c02a6246bd0",'
                      ' "sequence": 1, "at": "2026-09-13T06:11:54.510338+00:00", "selection_config_hash'
                      '": "67e653ace5b9", "closed_at": "2026-09-13T06:11:54.553800+00:00"}], "exposure_'
                      'event_ids": ["b88fe6fe13a9"]}'),
        'log': [
            ('{"at": "2026-09-13T06:11:54.510338+00:00", "authorisation": "prospective selecti'
             'on freeze", "call_index_base": 20260818, "event_id": "b88fe6fe13a9", "freeze_id"'
             ': "8c02a6246bd0", "k": 9, "n_seeds": 3, "reason": "fresh probe call on the holdo'
             'ut sample", "selection_config_hash": "67e653ace5b9", "sequence": 1, "strategy": '
             '"stratified", "type": "open"}'),
            ('{"at": "2026-09-13T06:11:54.553800+00:00", "event_id": "b88fe6fe13a9", "outcome"'
             ': "holdout benchmark completed on 1 meetings (replayed from committed cache)", "'
             'type": "close"}'),
        ],
    },
    'fb152f3: interrupted': {
        'ids': {'first': '73203858142b'},
        'selection': ('{"freeze_id": "4dd4c0fdc849", "config_hash": "67e653ace5b9", "strategy": "strati'
                      'fied", "k": 9, "n_seeds": 3, "call_index_base": 20260818, "max_output_tokens": 8'
                      '000, "frozen_at": "2026-09-13T06:11:54.576986+00:00", "authorised_exposures": 1,'
                      ' "authorisation": "prospective selection freeze", "note": "", "history": [], "re'
                      'validated": false, "prior_exposure_declared": false, "exposure_number": 1, "expo'
                      'sure_log": "replay_exposures.jsonl", "exposures": [{"at": "2026-09-13T06:11:54.5'
                      '91585+00:00", "authorisation": "prospective selection freeze", "call_index_base"'
                      ': 20260818, "event_id": "73203858142b", "freeze_id": "4dd4c0fdc849", "k": 9, "n_'
                      'seeds": 3, "reason": "fresh probe call on the holdout sample", "selection_config'
                      '_hash": "67e653ace5b9", "sequence": 1, "strategy": "stratified", "type": "open"}'
                      '], "exposure_stamps": [{"event_id": "73203858142b", "freeze_id": "4dd4c0fdc849",'
                      ' "sequence": 1, "at": "2026-09-13T06:11:54.591585+00:00", "selection_config_hash'
                      '": "67e653ace5b9", "closed_at": null}], "exposure_event_ids": ["73203858142b"]}'),
        'log': [
            ('{"at": "2026-09-13T06:11:54.591585+00:00", "authorisation": "prospective selecti'
             'on freeze", "call_index_base": 20260818, "event_id": "73203858142b", "freeze_id"'
             ': "4dd4c0fdc849", "k": 9, "n_seeds": 3, "reason": "fresh probe call on the holdo'
             'ut sample", "selection_config_hash": "67e653ace5b9", "sequence": 1, "strategy": '
             '"stratified", "type": "open"}'),
        ],
    },
    '62eb7ed: completed': {
        'ids': {'first': '42fe4c673286'},
        'selection': ('{"freeze_id": "f346089ccf52", "config_hash": "67e653ace5b9", "strategy": "strati'
                      'fied", "k": 9, "n_seeds": 3, "call_index_base": 20260818, "max_output_tokens": 8'
                      '000, "frozen_at": "2026-09-13T06:11:54.645227+00:00", "authorised_exposures": 1,'
                      ' "authorisation": "prospective selection freeze", "note": "", "history": [], "re'
                      'validated": false, "prior_exposure_declared": false, "exposure_number": 1, "expo'
                      'sure_log": "replay_exposures.jsonl", "exposures": [{"at": "2026-09-13T06:11:54.6'
                      '61663+00:00", "authorisation": "prospective selection freeze", "call_index_base"'
                      ': 20260818, "event_id": "42fe4c673286", "freeze_id": "f346089ccf52", "k": 9, "n_'
                      'seeds": 3, "reason": "fresh probe call on the holdout sample", "selection_config'
                      '_hash": "67e653ace5b9", "sequence": 1, "strategy": "stratified", "type": "open"}'
                      '], "exposure_stamps": [{"event_id": "42fe4c673286", "freeze_id": "f346089ccf52",'
                      ' "sequence": 1, "at": "2026-09-13T06:11:54.661663+00:00", "selection_config_hash'
                      '": "67e653ace5b9", "closed_at": "2026-09-13T06:11:54.698463+00:00"}], "exposure_'
                      'event_ids": ["42fe4c673286"]}'),
        'log': [
            ('{"at": "2026-09-13T06:11:54.661663+00:00", "authorisation": "prospective selecti'
             'on freeze", "call_index_base": 20260818, "event_id": "42fe4c673286", "freeze_id"'
             ': "f346089ccf52", "k": 9, "n_seeds": 3, "reason": "fresh probe call on the holdo'
             'ut sample", "selection_config_hash": "67e653ace5b9", "sequence": 1, "strategy": '
             '"stratified", "type": "open"}'),
            ('{"at": "2026-09-13T06:11:54.698463+00:00", "event_id": "42fe4c673286", "fresh": '
             'true, "outcome": "holdout benchmark completed on 1 meetings", "type": "close"}'),
        ],
    },
    '62eb7ed: completed from cache': {
        'ids': {'first': '9fdf3ff1e638'},
        'selection': ('{"freeze_id": "95a9466f52de", "config_hash": "67e653ace5b9", "strategy": "strati'
                      'fied", "k": 9, "n_seeds": 3, "call_index_base": 20260818, "max_output_tokens": 8'
                      '000, "frozen_at": "2026-09-13T06:11:54.720471+00:00", "authorised_exposures": 1,'
                      ' "authorisation": "prospective selection freeze", "note": "", "history": [], "re'
                      'validated": false, "prior_exposure_declared": false, "exposure_number": 1, "expo'
                      'sure_log": "replay_exposures.jsonl", "exposures": [{"at": "2026-09-13T06:11:54.7'
                      '38741+00:00", "authorisation": "prospective selection freeze", "call_index_base"'
                      ': 20260818, "event_id": "9fdf3ff1e638", "freeze_id": "95a9466f52de", "k": 9, "n_'
                      'seeds": 3, "reason": "fresh probe call on the holdout sample", "selection_config'
                      '_hash": "67e653ace5b9", "sequence": 1, "strategy": "stratified", "type": "open"}'
                      '], "exposure_stamps": [{"event_id": "9fdf3ff1e638", "freeze_id": "95a9466f52de",'
                      ' "sequence": 1, "at": "2026-09-13T06:11:54.738741+00:00", "selection_config_hash'
                      '": "67e653ace5b9", "closed_at": "2026-09-13T06:11:54.783212+00:00"}], "exposure_'
                      'event_ids": ["9fdf3ff1e638"]}'),
        'log': [
            ('{"at": "2026-09-13T06:11:54.738741+00:00", "authorisation": "prospective selecti'
             'on freeze", "call_index_base": 20260818, "event_id": "9fdf3ff1e638", "freeze_id"'
             ': "95a9466f52de", "k": 9, "n_seeds": 3, "reason": "fresh probe call on the holdo'
             'ut sample", "selection_config_hash": "67e653ace5b9", "sequence": 1, "strategy": '
             '"stratified", "type": "open"}'),
            ('{"at": "2026-09-13T06:11:54.783212+00:00", "event_id": "9fdf3ff1e638", "fresh": '
             'false, "outcome": "holdout benchmark completed on 1 meetings (replayed from comm'
             'itted cache)", "type": "close"}'),
        ],
    },
    '62eb7ed: interrupted': {
        'ids': {'first': '8ce1164c9f97'},
        'selection': ('{"freeze_id": "59050fa2cf39", "config_hash": "67e653ace5b9", "strategy": "strati'
                      'fied", "k": 9, "n_seeds": 3, "call_index_base": 20260818, "max_output_tokens": 8'
                      '000, "frozen_at": "2026-09-13T06:11:54.804581+00:00", "authorised_exposures": 1,'
                      ' "authorisation": "prospective selection freeze", "note": "", "history": [], "re'
                      'validated": false, "prior_exposure_declared": false, "exposure_number": 1, "expo'
                      'sure_log": "replay_exposures.jsonl", "exposures": [{"at": "2026-09-13T06:11:54.8'
                      '20601+00:00", "authorisation": "prospective selection freeze", "call_index_base"'
                      ': 20260818, "event_id": "8ce1164c9f97", "freeze_id": "59050fa2cf39", "k": 9, "n_'
                      'seeds": 3, "reason": "fresh probe call on the holdout sample", "selection_config'
                      '_hash": "67e653ace5b9", "sequence": 1, "strategy": "stratified", "type": "open"}'
                      '], "exposure_stamps": [{"event_id": "8ce1164c9f97", "freeze_id": "59050fa2cf39",'
                      ' "sequence": 1, "at": "2026-09-13T06:11:54.820601+00:00", "selection_config_hash'
                      '": "67e653ace5b9", "closed_at": null}], "exposure_event_ids": ["8ce1164c9f97"]}'),
        'log': [
            ('{"at": "2026-09-13T06:11:54.820601+00:00", "authorisation": "prospective selecti'
             'on freeze", "call_index_base": 20260818, "event_id": "8ce1164c9f97", "freeze_id"'
             ': "59050fa2cf39", "k": 9, "n_seeds": 3, "reason": "fresh probe call on the holdo'
             'ut sample", "selection_config_hash": "67e653ace5b9", "sequence": 1, "strategy": '
             '"stratified", "type": "open"}'),
        ],
    },
}


def _released(name, monkeypatch):
    """Write a Replay workspace exactly as a released version wrote it."""
    fixture = _RELEASED_WORKSPACES.get(name) or _RELEASED_CRASH_WORKSPACES[name]
    selection = json.loads(fixture["selection"])
    dr.SELECTION_STAMP.write_bytes(json.dumps(selection, indent=1).encode("utf-8"))
    dr.EXPOSURE_LOG.write_bytes("".join(line + "\n" for line in fixture["log"]).encode("utf-8"))
    monkeypatch.setattr(dr, "selection_config_hash", lambda: selection["config_hash"])
    return dict(fixture["ids"])


_RELEASED_STATES = [(writer, state) for writer in ("e91cdbc", "fb152f3", "62eb7ed")
                    for state in ("completed", "completed from cache", "interrupted")]


@pytest.mark.parametrize("writer,state", _RELEASED_STATES,
                         ids=[f"{writer}, {state}" for writer, state in _RELEASED_STATES])
def test_workspaces_written_by_released_versions_keep_ids_counts_and_permissions(
        cached_holdout, monkeypatch, tmp_path, writer, state):
    """
    Each workspace keeps its exposure id, count and allowance, and its benchmark replays and
    submits. Only an interrupted exposure whose selection recorded it as unfinished resumes:
    e91cdbc kept no completion state, so its interrupted exposure is replayed, never resumed.
    """
    w = cached_holdout
    ids = _released(f"{writer}: {state}", monkeypatch)
    dr.require_readable_log()
    assert dr.known_exposure_count() == 1
    assert dr._read_selection()["authorised_exposures"] == 1
    assert dr.originating_exposure_id() == ids["first"]

    event, sent = _holdout_request(monkeypatch, tmp_path, "a request under this release")
    if state == "interrupted" and writer != "e91cdbc":
        assert not isinstance(event, Exception) and event["resumed"], event
        assert event["event_id"] == ids["first"] and sent == 1
        dr.close_holdout_exposure(ids["first"], "benchmark completed", fresh=True)
    else:
        assert isinstance(event, RuntimeError) and sent == 0, (event, sent)

    replayed = w.benchmark()
    assert replayed.attrs["exposure_event_id"] == ids["first"] and w.requests["n"] == 0
    w.submit(w.report(replayed))
    assert dr.known_exposure_count() == 1
    assert dr._read_selection()["authorised_exposures"] == 1


def test_a_declared_damaged_line_stays_covered_when_restored_records_move_it(cached_holdout):
    """
    FOUND BY THE RESTORATION-ORDER LIFECYCLES (recheck-9). A declaration covered a damaged line
    by its number as well as its bytes, so restoring earlier records ahead of it moved the line
    and the same declared damage read as new: the runtime refused and submission failed. It now
    covers those bytes, that many times, wherever the line sits - and a further copy of the same
    bytes is still new damage.
    """
    w = cached_holdout
    dr.freeze_selection()
    w.expose("the first look")
    saved = dr.EXPOSURE_LOG.read_bytes()
    dr.EXPOSURE_LOG.unlink()
    _declare()
    dr.freeze_selection(revalidate=True, note="synthetic: a declared second look")
    second = w.expose("the second look")
    with dr.EXPOSURE_LOG.open("ab") as fh:
        fh.write(_DAMAGED_LINE)
    _declare("synthetic: a line of the new log was damaged")

    dr.EXPOSURE_LOG.write_bytes(saved + dr.EXPOSURE_LOG.read_bytes())
    dr.require_readable_log()
    w.submit(_report_naming(second))
    assert dr.known_exposure_count() == 2

    with dr.EXPOSURE_LOG.open("ab") as fh:
        fh.write(_DAMAGED_LINE)
    with pytest.raises(RuntimeError, match="damaged"):
        dr.require_readable_log()
    with pytest.raises(AssertionError, match="damaged"):
        w.submit(_report_naming(second))


# -------------------------------------------------------------------------------------------
# WHETHER DAMAGE COULD BE A COMPLETION RECORD IS DECIDED BY WHEN IT WAS DECLARED  (recheck-10 W1)
# -------------------------------------------------------------------------------------------
# A benchmark finished; the run crashed before the selection recorded it; the log lost the open
# record and the close became unreadable; the loss was declared; the open record was restored.
# Whether surviving damage could be the completion record was decided by where the damaged line
# sat relative to the open record, so restoring the open record AFTER the damage made the
# finished benchmark resumable: a real cache miss sent a request under the used exposure, and
# submission accepted it. Restored records move lines; they cannot move the time the damage was
# declared, which is what decides it now. Requests are counted at the scripted network.

_RELEASED_CRASH_WORKSPACES = {
    'b825b4e: completion logged, selection update lost in a crash': {
        'ids': {'first': '384429073f9d'},
        'selection': ('{"freeze_id": "a7c64d4632d3", "config_hash": "67e653ace5b9", "strategy": "strati'
                      'fied", "k": 9, "n_seeds": 3, "call_index_base": 20260818, "max_output_tokens": 8'
                      '000, "frozen_at": "2026-09-13T07:35:13.977501+00:00", "authorised_exposures": 1,'
                      ' "authorisation": "prospective selection freeze", "note": "", "history": [], "re'
                      'validated": false, "prior_exposure_declared": false, "exposure_number": 1, "expo'
                      'sure_log": "replay_exposures.jsonl", "exposures": [{"at": "2026-09-13T07:35:14.9'
                      '30390+00:00", "authorisation": "prospective selection freeze", "call_index_base"'
                      ': 20260818, "event_id": "384429073f9d", "freeze_id": "a7c64d4632d3", "k": 9, "n_'
                      'seeds": 3, "reason": "fresh probe call on the holdout sample", "selection_config'
                      '_hash": "67e653ace5b9", "sequence": 1, "strategy": "stratified", "type": "open"}'
                      '], "exposure_stamps": [{"event_id": "384429073f9d", "freeze_id": "a7c64d4632d3",'
                      ' "sequence": 1, "at": "2026-09-13T07:35:14.930390+00:00", "selection_config_hash'
                      '": "67e653ace5b9", "closed_at": null}], "exposure_event_ids": ["384429073f9d"]}'),
        'log': [
            ('{"at": "2026-09-13T07:35:14.930390+00:00", "authorisation": "prospective selecti'
             'on freeze", "call_index_base": 20260818, "event_id": "384429073f9d", "freeze_id"'
             ': "a7c64d4632d3", "k": 9, "n_seeds": 3, "reason": "fresh probe call on the holdo'
             'ut sample", "selection_config_hash": "67e653ace5b9", "sequence": 1, "strategy": '
             '"stratified", "type": "open"}'),
            ('{"at": "2026-09-13T07:35:15.685176+00:00", "event_id": "384429073f9d", "fresh": '
             'true, "outcome": "holdout benchmark completed on 1 meetings", "type": "close"}'),
        ],
    },
}


_W1_LAYOUTS = ("before the damage", "after the declaration", "twice, around the damage",
               "between the damage and the declaration")


def _w1_layouts(open_line, declared, close_line=None):
    """The restored open record placed before, after, twice around and inside the surviving
    damaged evidence and its declaration - the same lines in different physical orders."""
    head, _, rest = declared.partition(b"\n")
    layouts = {"before the damage": open_line + declared,
               "after the declaration": declared + open_line,
               "twice, around the damage": open_line + declared + open_line,
               "between the damage and the declaration": head + b"\n" + open_line + rest}
    if close_line is not None:
        layouts["with its readable close"] = open_line + close_line + declared
    return layouts


def _completion_crash(monkeypatch, tmp_path, question="the original question"):
    """Run the real benchmark and fail only the selection write that records its completion.
    Returns (the exposure id, its open line, the cached answers)."""
    real = config.atomic_write_text

    def completion_write_fails(path, text, *args, **kwargs):
        if pathlib.Path(path) == pathlib.Path(dr.SELECTION_STAMP) and any(
                s.get("closed_at") for s in json.loads(text).get("exposure_stamps") or []):
            raise _Crash("synthetic crash while the selection recorded completion")
        return real(path, text, *args, **kwargs)

    monkeypatch.setattr(config, "atomic_write_text", completion_write_fails)
    crashed, sent = _run_benchmark(monkeypatch, tmp_path, question)
    monkeypatch.setattr(config, "atomic_write_text", real)
    assert isinstance(crashed, _Crash) and sent == 1, (crashed, sent)
    stamp = dr._read_selection()["exposure_stamps"][0]
    assert stamp["closed_at"] is None, "the crash must leave the completion unrecorded"
    lines = dr.EXPOSURE_LOG.read_text(encoding="utf-8").splitlines()
    open_line = next(line for line in lines
                     if json.loads(line).get("type") == "open").encode("utf-8") + b"\n"
    cache = {p: p.read_bytes() for p in (tmp_path / "benchmark-raw").glob("*.json")}
    assert len(cache) == 1
    return stamp["event_id"], open_line, cache


def _unreadable_completion(event_id, damage):
    """What is left of a completion record that can no longer be read."""
    if damage == "malformed":
        return b"{synthetic damaged completion record\n"
    close = json.dumps({"at": "2026-09-13T00:00:00+00:00", "event_id": event_id,
                        "fresh": True, "outcome": "holdout benchmark completed on 1 meetings",
                        "type": "close"}, sort_keys=True).encode("utf-8")
    return close[: len(close) // 2]


@pytest.mark.parametrize("layout", _W1_LAYOUTS)
@pytest.mark.parametrize("damage", ["malformed", "truncated"])
def test_restoring_an_open_record_after_damage_never_reopens_a_finished_benchmark(
        cached_holdout, monkeypatch, tmp_path, damage, layout):
    """
    THE DEFECT THIS PINS (W1), through this release's writer: the completion-write crash, the
    damage, the declaration and the restored open record in every position. Whatever the order,
    a fresh cache miss is refused at allowance one - here and in a fresh process - and leaves no
    answer in the cache; the original answer replays under the same exposure with no request;
    an authorised look is a new exposure.
    """
    w = cached_holdout
    dr.freeze_selection()
    first, open_line, cache = _completion_crash(monkeypatch, tmp_path)
    dr.EXPOSURE_LOG.write_bytes(_unreadable_completion(first, damage))
    _declare()
    dr.EXPOSURE_LOG.write_bytes(_w1_layouts(open_line, dr.EXPOSURE_LOG.read_bytes())[layout])
    dr.require_readable_log()
    assert dr.known_exposure_count() == 1
    assert dr._read_selection()["authorised_exposures"] == 1
    assert dr.open_exposure(dr._read_selection()["freeze_id"]) is None

    for path in cache:
        path.unlink()
    refused, sent = _run_benchmark(monkeypatch, tmp_path, "the original question")
    assert isinstance(refused, RuntimeError) and sent == 0, (refused, sent)
    assert not list((tmp_path / "benchmark-raw").glob("*.json")), (
        "a refused request must leave no answer in the cache")
    restarted = _restart(tmp_path)
    assert restarted["outcome"] == "refused" and restarted["requests"] == 0, restarted

    for path, data in cache.items():
        path.write_bytes(data)
    recovered, sent = _run_benchmark(monkeypatch, tmp_path, "the original question")
    assert not isinstance(recovered, Exception), recovered
    assert sent == 0 and not recovered.attrs["exposure_was_fresh"]
    assert recovered.attrs["exposure_event_id"] == first
    w.submit(w.report(recovered))

    dr.freeze_selection(revalidate=True, note="synthetic: a declared second look")
    for path in cache:
        path.unlink()
    second, sent = _run_benchmark(monkeypatch, tmp_path, "the original question")
    assert not isinstance(second, Exception), second
    assert sent == 1 and second.attrs["exposure_was_fresh"]
    assert second.attrs["exposure_event_id"] != first
    assert dr.known_exposure_count() == 2
    assert dr._read_selection()["authorised_exposures"] == 2
    w.submit(w.report(second))


@pytest.mark.parametrize("layout", _W1_LAYOUTS + ("with its readable close",))
@pytest.mark.parametrize("damage", ["malformed", "truncated"])
def test_a_crash_state_left_by_the_log_first_release_is_never_reopened_after_damage(
        cached_holdout, monkeypatch, tmp_path, damage, layout):
    """
    The same recovery from the files the released b825b4e writer left after a crash between its
    close line and its selection update - the order that opened the window. The open record
    restored anywhere relative to the declared damage never permits a fresh request; restoring
    the readable close as well is a clean recovery.
    """
    w = cached_holdout
    ids = _released("b825b4e: completion logged, selection update lost in a crash", monkeypatch)
    first = ids["first"]
    lines = w.lines()
    open_line = next(line for line in lines
                     if json.loads(line).get("type") == "open").encode("utf-8") + b"\n"
    close_line = next(line for line in lines
                      if json.loads(line).get("type") == "close").encode("utf-8") + b"\n"
    assert dr._read_selection()["exposure_stamps"][0]["closed_at"] is None
    damaged = (b"{synthetic damaged completion record\n" if damage == "malformed"
               else close_line[: len(close_line) // 2])
    dr.EXPOSURE_LOG.write_bytes(damaged)
    _declare()
    dr.EXPOSURE_LOG.write_bytes(
        _w1_layouts(open_line, dr.EXPOSURE_LOG.read_bytes(), close_line)[layout])
    dr.require_readable_log()
    assert dr.known_exposure_count() == 1
    assert dr._read_selection()["authorised_exposures"] == 1

    refused, sent = _holdout_request(monkeypatch, tmp_path, "a question after the restoration")
    assert isinstance(refused, RuntimeError) and sent == 0, (refused, sent)
    restarted = _restart(tmp_path)
    assert restarted["outcome"] == "refused" and restarted["requests"] == 0, restarted

    replayed = w.benchmark()
    assert replayed.attrs["exposure_event_id"] == first and w.requests["n"] == 0
    w.submit(w.report(replayed))
    dr.freeze_selection(revalidate=True, note="synthetic: a declared second look")
    second, sent = _holdout_request(monkeypatch, tmp_path, "the declared second look")
    assert not isinstance(second, Exception) and not second["resumed"], second
    assert sent == 1 and second["sequence"] == 2 and second["event_id"] != first


@pytest.mark.parametrize("moved", [False, True],
                         ids=["damage where it was written", "damage moved after the new exposure"])
def test_an_exposure_begun_after_older_declared_damage_still_resumes(cached_holdout, monkeypatch,
                                                                      tmp_path, moved):
    """
    The control for W1: damage declared before an exposure began cannot be its completion
    record - wherever the damaged line ends up - so an interrupted latest exposure created after
    it resumes under its own id, in this process and in a fresh one.
    """
    w = cached_holdout
    dr.freeze_selection()
    w.expose("the first look")
    with dr.EXPOSURE_LOG.open("ab") as fh:
        fh.write(_DAMAGED_LINE)
    _declare()
    dr.freeze_selection(revalidate=True, note="synthetic: a declared second look")
    second, sent = _holdout_request(monkeypatch, tmp_path, "the second look, interrupted")
    assert sent == 1 and not second["resumed"]
    if moved:
        lines = w.lines()
        w.write([line for line in lines if not line.startswith("{synthetic damaged line")]
                + [line for line in lines if line.startswith("{synthetic damaged line")])
    resumed, sent = _holdout_request(monkeypatch, tmp_path, "the second look, resumed")
    assert not isinstance(resumed, Exception), resumed
    assert resumed["resumed"] and resumed["event_id"] == second["event_id"] and sent == 1
    restarted = _restart(tmp_path)
    assert restarted["outcome"] == "resumed" and restarted["requests"] == 1, restarted
    assert restarted["event_id"] == second["event_id"] and restarted["known"] == 2, restarted


def _unparsable(line):
    try:
        return not isinstance(json.loads(line.decode("utf-8")), dict)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return True


_REORDERINGS = {
    "reversed": lambda lines: lines[::-1],
    "rotated": lambda lines: lines[1:] + lines[:1],
    "damage first": lambda lines: ([l for l in lines if _unparsable(l)]
                                   + [l for l in lines if not _unparsable(l)]),
    "damage last": lambda lines: ([l for l in lines if not _unparsable(l)]
                                  + [l for l in lines if _unparsable(l)]),
}


def _sampling_permissions():
    rec = dr._read_selection()
    event, _ = dr.unfinished_exposure(rec["freeze_id"], rec)
    return {"resumable": (event or {}).get("event_id"),
            "known": dr.known_exposure_count(),
            "history problems": len(dr.exposure_history_problems(rec, dr.exposure_events())),
            "origin": dr.originating_exposure_id()}


def _history_completion_crash(w, monkeypatch, tmp_path):
    dr.freeze_selection()
    first, open_line, _ = _completion_crash(monkeypatch, tmp_path)
    dr.EXPOSURE_LOG.write_bytes(_unreadable_completion(first, "malformed"))
    _declare()
    dr.EXPOSURE_LOG.write_bytes(dr.EXPOSURE_LOG.read_bytes() + open_line)


def _history_older_damage(w, monkeypatch, tmp_path):
    dr.freeze_selection()
    w.expose("the first look")
    with dr.EXPOSURE_LOG.open("ab") as fh:
        fh.write(_DAMAGED_LINE)
    _declare()
    dr.freeze_selection(revalidate=True, note="synthetic: a declared second look")
    _holdout_request(monkeypatch, tmp_path, "the second look, interrupted")


def _history_superseded(w, monkeypatch, tmp_path):
    dr.freeze_selection()
    _holdout_request(monkeypatch, tmp_path, "the first look, interrupted")
    saved = dr.EXPOSURE_LOG.read_bytes()
    dr.EXPOSURE_LOG.unlink()
    _declare()
    dr.freeze_selection(revalidate=True, note="synthetic: a declared replacement look")
    w.expose("the replacement")
    dr.EXPOSURE_LOG.write_bytes(saved + dr.EXPOSURE_LOG.read_bytes())


_ORDER_HISTORIES = {
    "a completion lost in a crash, damaged, and its open record restored":
        _history_completion_crash,
    "an interrupted exposure begun after older declared damage": _history_older_damage,
    "a superseded exposure's records restored": _history_superseded,
}


@pytest.mark.parametrize("history", list(_ORDER_HISTORIES))
def test_the_order_of_identical_records_never_changes_what_may_be_sampled(
        cached_holdout, monkeypatch, tmp_path, history):
    """
    THE INVARIANT W1 BROKE: the same historical records in a different physical order must not
    grant or withdraw sampling permission. Every reordering gives the same answers - which
    exposure may resume, how many exposures count, whether the history is valid, and which
    exposure the benchmark came from.
    """
    w = cached_holdout
    _ORDER_HISTORIES[history](w, monkeypatch, tmp_path)
    baseline = dr.EXPOSURE_LOG.read_bytes()
    expected = _sampling_permissions()
    lines = [line for line in baseline.split(b"\n") if line.strip()]
    for name, reorder in _REORDERINGS.items():
        dr.EXPOSURE_LOG.write_bytes(b"\n".join(reorder(lines)) + b"\n")
        assert _sampling_permissions() == expected, f"{history}, {name}"
    dr.EXPOSURE_LOG.write_bytes(baseline)
