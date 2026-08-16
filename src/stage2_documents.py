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

# Retrieval query: the concepts that drive a policy stance judgement.
RETRIEVAL_QUERY = (
    "inflation outlook and forecasts; labour market conditions and wages; "
    "the cash rate decision and considerations; risks to the outlook; "
    "policy stance whether restrictive or accommodative"
)
TOP_K = 8


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


def split_paragraphs(text: str, min_chars: int = 220) -> list[str]:
    """RBA minutes are one long run of sentences after HTML stripping, so chunk on
    sentence boundaries and glue short fragments onto the previous chunk."""
    sents = re.split(r"(?<=[.!?])\s+", text)
    chunks, cur = [], ""
    for s in sents:
        cur = (cur + " " + s).strip()
        if len(cur) >= min_chars:
            chunks.append(cur)
            cur = ""
    if cur:
        if chunks:
            chunks[-1] += " " + cur
        else:
            chunks.append(cur)
    return chunks


def run() -> pd.DataFrame:
    from sentence_transformers import SentenceTransformer

    files = sorted(glob.glob(str(config.RBA_MINUTES_DIR / "*.html")))
    if not files:
        raise RuntimeError(f"No minutes found in {config.RBA_MINUTES_DIR}")

    model = SentenceTransformer(config.EMBEDDING_MODEL)
    q_emb = model.encode([RETRIEVAL_QUERY], normalize_embeddings=True)[0]

    rows = []
    for i, f in enumerate(files, 1):
        date = re.search(r"(\d{4}-\d{2}-\d{2})", f).group(1)
        full = extract_text(f)
        chunks = split_paragraphs(full)
        if not chunks:
            print(f"  WARNING: no text extracted from {f}")
            continue
        emb = model.encode(chunks, normalize_embeddings=True, show_progress_bar=False)
        sims = emb @ q_emb
        top = np.argsort(-sims)[:TOP_K]
        top = sorted(top)  # keep document order so the retrieved text reads coherently
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
