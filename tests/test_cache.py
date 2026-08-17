"""Cache invalidation tests for stage 3.

None of these call the API. `score_doc` is monkeypatched, so they run offline and fast.

Each guards a failure that produced NO error and a wrong result:

  1. Changing retrieval settings reused answers scored from the OLD passages, because the
     fingerprint did not cover retrieval and the filename was just the meeting date.
  2. A run rate-limited into 3 of 10 successful calls was cached as complete and reused
     forever, so the uncertainty layer measured spread across 3 calls while the manifest
     claimed 10.
  3. A crash during json.dump left a truncated file that the next run parsed as a short cache.
"""
from __future__ import annotations
import json

import pytest

from src import stage3_riskvoice as s3


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    """Point the cache at a temp dir so tests never touch data/processed/llm_raw."""
    monkeypatch.setattr(s3.config, "LLM_RAW", tmp_path)
    return tmp_path


def _fake_calls(n, tag="ok", seed_offset=0):
    return [{f: 0.5 for f in s3.FIELD_DESCRIPTIONS} | {"_seed": 1000 + seed_offset + i,
                                                       "_tag": tag}
            for i in range(n)]


class TestFingerprintCoverage:
    def test_prompt_change_changes_fingerprint(self, monkeypatch):
        before = s3._settings_fingerprint()
        monkeypatch.setattr(s3, "SYSTEM_PROMPT", s3.SYSTEM_PROMPT + " extra")
        assert s3._settings_fingerprint() != before

    def test_retrieval_query_change_changes_fingerprint(self, monkeypatch):
        """THE BUG: retrieval was not in the key, so changing it reused stale scores."""
        from src import stage2_documents as s2
        before = s3._settings_fingerprint()
        monkeypatch.setattr(s2, "RETRIEVAL_QUERY", s2.RETRIEVAL_QUERY + " funding costs")
        assert s3._settings_fingerprint() != before

    def test_top_k_change_changes_fingerprint(self, monkeypatch):
        from src import stage2_documents as s2
        before = s3._settings_fingerprint()
        monkeypatch.setattr(s2, "TOP_K", s2.TOP_K + 1)
        assert s3._settings_fingerprint() != before

    def test_text_column_change_changes_fingerprint(self, monkeypatch):
        before = s3._settings_fingerprint()
        monkeypatch.setattr(s3, "TEXT_COLUMN", "text_full")
        assert s3._settings_fingerprint() != before

    def test_n_calls_change_changes_fingerprint(self, monkeypatch):
        before = s3._settings_fingerprint()
        monkeypatch.setattr(s3.config, "N_PARALLEL_CALLS", s3.config.N_PARALLEL_CALLS + 1)
        assert s3._settings_fingerprint() != before

    def test_seed_change_changes_fingerprint(self, monkeypatch):
        before = s3._settings_fingerprint()
        monkeypatch.setattr(s3.config, "SEED", s3.config.SEED + 1)
        assert s3._settings_fingerprint() != before

    def test_schema_version_change_changes_fingerprint(self, monkeypatch):
        before = s3._settings_fingerprint()
        monkeypatch.setattr(s3, "SCHEMA_VERSION", s3.SCHEMA_VERSION + 1)
        assert s3._settings_fingerprint() != before

    def test_identical_settings_are_stable(self):
        assert s3._settings_fingerprint() == s3._settings_fingerprint()


class TestDocumentTextInPath:
    def test_different_text_gives_different_cache_file(self):
        """Same meeting date, different retrieved passages: must not collide."""
        a = s3._cache_path("2011-08-02", "passage A")
        b = s3._cache_path("2011-08-02", "passage B")
        assert a != b

    def test_same_text_gives_the_same_file(self):
        assert s3._cache_path("2011-08-02", "same") == s3._cache_path("2011-08-02", "same")

    def test_date_is_still_visible_in_the_filename(self):
        assert "2011-08-02" in s3._cache_path("2011-08-02", "x").name


class TestTopUpAndReuse:
    def test_complete_cache_is_reused_without_calling_the_api(self, monkeypatch):
        def boom(*a, **k):
            raise AssertionError("score_doc must not be called when the cache is complete")
        monkeypatch.setattr(s3, "score_doc", boom)
        p = s3._cache_path("2020-03-03", "txt")
        s3._write_atomic(p, _fake_calls(10))
        calls, cached = s3.cached_score_doc("txt", "2020-03-03", n=10)
        assert cached and len(calls) == 10

    def test_short_cache_is_topped_up_not_accepted(self, monkeypatch):
        """THE BUG: 3 of 10 calls was treated as a complete cached run."""
        made = {}
        def fake(text, n=None, seed_offset=0):
            made["n"] = n
            made["offset"] = seed_offset
            return _fake_calls(n, tag="new", seed_offset=seed_offset)
        monkeypatch.setattr(s3, "score_doc", fake)

        p = s3._cache_path("2020-03-03", "txt")
        s3._write_atomic(p, _fake_calls(3, tag="old"))
        calls, cached = s3.cached_score_doc("txt", "2020-03-03", n=10)

        assert made["n"] == 7, "should buy only the missing calls"
        assert not cached
        assert len(calls) == 10
        assert sum(c["_tag"] == "old" for c in calls) == 3

    def test_topped_up_calls_use_fresh_seeds(self, monkeypatch):
        monkeypatch.setattr(s3, "score_doc",
                            lambda text, n=None, seed_offset=0:
                            _fake_calls(n, "new", seed_offset))
        p = s3._cache_path("2020-03-03", "txt")
        s3._write_atomic(p, _fake_calls(3, "old"))
        calls, _ = s3.cached_score_doc("txt", "2020-03-03", n=10)
        seeds = [c["_seed"] for c in calls]
        assert len(set(seeds)) == len(seeds), "top-up reused seeds already spent"

    def test_top_up_is_persisted(self, monkeypatch):
        monkeypatch.setattr(s3, "score_doc",
                            lambda text, n=None, seed_offset=0:
                            _fake_calls(n, "new", seed_offset))
        p = s3._cache_path("2020-03-03", "txt")
        s3._write_atomic(p, _fake_calls(3, "old"))
        s3.cached_score_doc("txt", "2020-03-03", n=10)
        assert len(json.load(open(p))) == 10

    def test_error_rows_do_not_count_as_valid_calls(self, monkeypatch):
        monkeypatch.setattr(s3, "score_doc",
                            lambda text, n=None, seed_offset=0:
                            _fake_calls(n, "new", seed_offset))
        p = s3._cache_path("2020-03-03", "txt")
        s3._write_atomic(p, _fake_calls(2) + [{"_error": "rate limit"}] * 8)
        calls, cached = s3.cached_score_doc("txt", "2020-03-03", n=10)
        assert not cached and len(calls) == 10
        assert all("_error" not in c for c in calls)


class TestCorruptCache:
    def test_truncated_json_is_re_run_not_crashed_on(self, monkeypatch):
        monkeypatch.setattr(s3, "score_doc",
                            lambda text, n=None, seed_offset=0: _fake_calls(n))
        p = s3._cache_path("2020-03-03", "txt")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text('[{"financial_conditions_conc')      # crash mid-write
        calls, cached = s3.cached_score_doc("txt", "2020-03-03", n=10)
        assert not cached and len(calls) == 10

    def test_atomic_write_leaves_no_temp_file(self):
        p = s3._cache_path("2020-03-03", "txt")
        s3._write_atomic(p, _fake_calls(2))
        assert p.exists()
        assert not p.with_suffix(".tmp").exists()
