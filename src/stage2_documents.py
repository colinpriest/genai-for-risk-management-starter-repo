"""
Stage 2 — Documents and retrieval

OWNER: solution author

Parses the RBA minutes HTML, extracts clean text, and builds a retrieval step that selects
the passages most relevant to monetary policy stance. Retrieval, not full-context stuffing.

WRITES data/processed/documents.parquet
"""
from __future__ import annotations
import re
import glob
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import config

# =============================================================================================
# YOUR CHOICE: what to retrieve.
#
# The single query below is written for a POLICY STANCE judgement - inflation, wages, the cash
# rate. That is the reference implementation's target and it is NOT what you are scoring. Used
# unchanged for all six fields, it biases every construct toward stance: passages about market
# funding or offshore risk score low on this query and never reach the model, so
# financial_conditions_concern and global_risk_salience are judged on text that was selected
# for a different purpose.
#
# Two options, and you must justify whichever you pick:
#   1. Rewrite RETRIEVAL_QUERY so it spans all five constructs.
#   2. Better: set PER_CONSTRUCT_QUERIES so each construct retrieves its own passages. The
#      union is then scored, so every construct has its evidence present.
#
# Validate whichever you choose: sample some documents, read what was retrieved, and check the
# passages you would have picked by hand are in there. Report the recall you found.
# =============================================================================================
RETRIEVAL_QUERY = (
    "inflation outlook and forecasts; labour market conditions and wages; "
    "the cash rate decision and considerations; risks to the outlook; "
    "policy stance whether restrictive or accommodative"
)

# Leave empty to use the single query above. Fill it in to retrieve per construct.
PER_CONSTRUCT_QUERIES: dict[str, str] = {
    # "financial_conditions_concern": "credit availability, funding costs, bank lending, "
    #                                 "housing finance, financial market conditions",
    # "global_risk_salience":         "offshore developments, international markets, global "
    #                                 "economy, foreign central banks, world growth",
    # ... add the rest
}

TOP_K = 8

# Chunk overlap, in sentences. Zero overlap splits arguments across a boundary, so a passage
# whose claim starts in one chunk and concludes in the next matches neither well.
CHUNK_OVERLAP_SENTENCES = 1


def extract_text(path: str) -> str:
    html = open(path, encoding="utf-8", errors="replace").read()
    soup = BeautifulSoup(html, "lxml")
    for t in soup(["script", "style", "nav", "header", "footer", "aside"]):
        t.decompose()
    main = soup.find("div", {"id": "content"}) or soup.find("main") or soup
    text = " ".join(main.get_text(" ").split())
    # Drop the trailing "Related Information" block - it is navigation, not content.
    text = re.split(r"\bRelated Information\b", text)[0]
    return text.strip()


def split_paragraphs(text: str, min_chars: int = 220,
                     overlap: int | None = None) -> list[str]:
    """Chunk on sentence boundaries, with OVERLAP between adjacent chunks.

    RBA minutes are one long run of sentences after HTML stripping. Chunking with no overlap
    cuts arguments in half: "members noted that funding costs had risen. This was expected to
    weigh on credit growth." split across a boundary leaves neither half matching a query
    about credit conditions. Repeating the last sentence of each chunk at the start of the
    next costs little and fixes it.

    min_chars is a character budget, not a token budget. It is a rough proxy - roughly four
    characters per token for English - and it is used because it needs no tokenizer. If you
    care about exact context budgeting, count tokens with tiktoken instead and say so.
    """
    overlap = CHUNK_OVERLAP_SENTENCES if overlap is None else overlap
    sents = [s for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    chunks, cur_sents, n_new = [], [], 0
    for s in sents:
        cur_sents.append(s)
        n_new += 1
        if len(" ".join(cur_sents)) >= min_chars:
            chunks.append(" ".join(cur_sents))
            cur_sents = cur_sents[-overlap:] if overlap else []
            n_new = 0
    # Only flush the tail if it contains sentences not already emitted; otherwise it is just
    # the carried-over overlap and appending it duplicates text.
    if n_new:
        tail = " ".join(cur_sents)
        if chunks:
            chunks[-1] += " " + " ".join(cur_sents[-n_new:])
        else:
            chunks.append(tail)
    return chunks


def run() -> pd.DataFrame:
    from sentence_transformers import SentenceTransformer

    files = sorted(glob.glob(str(config.RBA_MINUTES_DIR / "*.html")))
    if not files:
        raise RuntimeError(f"No minutes found in {config.RBA_MINUTES_DIR}")

    # CORPUS_START / CORPUS_END actually filter. They used to be declared and ignored, so
    # narrowing the window in config.py silently did nothing.
    def _in_window(path: str) -> bool:
        m = re.search(r"(\d{4}-\d{2}-\d{2})", path)
        if not m:
            return False
        d = pd.Timestamp(m.group(1))
        if config.CORPUS_START and d < pd.Timestamp(config.CORPUS_START):
            return False
        if config.CORPUS_END and d > pd.Timestamp(config.CORPUS_END):
            return False
        return True

    n_all = len(files)
    files = [f for f in files if _in_window(f)]
    print(f"  corpus window {config.CORPUS_START} to {config.CORPUS_END or 'latest'}: "
          f"{len(files)} of {n_all} documents")
    if not files:
        raise RuntimeError("Corpus window excluded every document - check CORPUS_START/END")

    model = SentenceTransformer(config.EMBEDDING_MODEL)
    if PER_CONSTRUCT_QUERIES:
        qs = list(PER_CONSTRUCT_QUERIES.values())
        print(f"  retrieving per construct: {len(qs)} queries, union of top-{TOP_K} each")
    else:
        qs = [RETRIEVAL_QUERY]
        print("  retrieving with ONE query written for policy stance - see the note in "
              "stage2_documents.py about the bias this introduces")
    q_embs = model.encode(qs, normalize_embeddings=True)
    q_emb = q_embs[0]

    rows = []
    for i, f in enumerate(files, 1):
        date = re.search(r"(\d{4}-\d{2}-\d{2})", f).group(1)
        full = extract_text(f)
        chunks = split_paragraphs(full)
        if not chunks:
            print(f"  WARNING: no text extracted from {f}")
            continue
        emb = model.encode(chunks, normalize_embeddings=True, show_progress_bar=False)
        all_sims = emb @ q_embs.T                     # (n_chunks, n_queries)
        # Union of each query's top-K, so every construct's evidence is present. With one
        # query this is exactly the old behaviour.
        picked = set()
        for qi in range(all_sims.shape[1]):
            picked.update(np.argsort(-all_sims[:, qi])[:TOP_K].tolist())
        top = sorted(picked)   # document order, so the retrieved text reads coherently
        sims = all_sims.max(axis=1)
        retrieved = " ".join(chunks[j] for j in top)
        rows.append({
            "meeting_date": pd.to_datetime(date),
            "n_chars_full": len(full),
            "n_chunks": len(chunks),
            "n_chars_retrieved": len(retrieved),
            "mean_top_sim": float(sims[top].mean()),
            "text_full": full,
            "text_retrieved": retrieved,
        })
        if i % 50 == 0:
            print(f"  {i}/{len(files)}")

    df = pd.DataFrame(rows).sort_values("meeting_date").reset_index(drop=True)
    df.to_parquet(config.DATA_PROCESSED / "documents.parquet", index=False)

    print(f"  documents: {len(df)} rows, {df.meeting_date.min().date()} -> {df.meeting_date.max().date()}")
    print(f"  full text chars   : median {df.n_chars_full.median():.0f}  min {df.n_chars_full.min()}  max {df.n_chars_full.max()}")
    print(f"  retrieved chars   : median {df.n_chars_retrieved.median():.0f}")
    print(f"  compression       : {df.n_chars_retrieved.sum()/df.n_chars_full.sum():.1%} of full text")
    print(f"  retrieval quality : mean top-{TOP_K} similarity {df.mean_top_sim.mean():.3f}")
    return df


if __name__ == "__main__":
    run()
