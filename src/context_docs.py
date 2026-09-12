"""
Load the Shock context documents you downloaded. SUPPLIED - do not modify.

WHY THE DOCUMENTS ARE NOT IN THIS REPOSITORY
    Several of the Shock sources - the WEF Global Risks Report, Eurasia Group Top Risks,
    the Lloyd's systemic risk scenarios - are free to read but NOT free to redistribute.
    The course cannot ship them to you. You download them yourself into

        data/raw/context/

    See data/raw/context/README.md for the list and the links, and
    draft-assignment/shock-context-sources.md for what each one is good for.

WHY THIS MATTERS FOR THE ANALYSIS, NOT JUST THE LICENCE
    The two scenarios are FIXED - you do not invent them. What these sources supply is the
    GLOBAL evidence leg: the passages your channels quote, verified verbatim and read by a
    person. A channel grounded in the model's general knowledge instead is barred.

    Not from the model's general knowledge, because a scenario an LLM invents unprompted is
    drawn from whatever was common in its training data, which is a poor sample of the risks
    that matter to Australia in 2026.

    Not from RBA documents, because the scenario is the CAUSE and the RBA's reaction is the
    EFFECT. A scenario taken from RBA minutes is a risk the Board has already identified and
    already responded to, so its effect is already inside your model and it cannot test
    anything.

WHATEVER YOU DOWNLOAD IS WHAT GETS USED
    This module does not look for particular filenames. It reads every supported file in
    the folder and uses what it finds, so different teams will legitimately be working from
    different sources - and your report should say which ones you used and why.

    You are expected to have at least three independent sources. One report's risk taxonomy
    is not a survey of geopolitical risk, it is one organisation's house view.

SUPPORTED FORMATS: .pdf, .html, .htm, .txt, .md
"""
from __future__ import annotations

import bisect
import json
import re
from datetime import datetime
from pathlib import Path

import config

CONTEXT_DIR = config.DATA_RAW / "context"
SUPPORTED = {".pdf", ".html", ".htm", ".txt", ".md"}

# Rough character budget for what gets pasted into a single prompt. The course model has a
# large context window but a 200-page PDF dumped whole produces worse reasoning, not better
# - the relevant passages get buried. Retrieval, not stuffing. The budget also keeps a
# single call inside the proxy's per-request token ceiling.
DEFAULT_BUDGET_CHARS = 24_000


# Character offset of each page start, per document, IN THE NORMALISED TEXT that retrieval
# actually searches. Getting this wrong is not cosmetic: it produces a citation to a real
# document and a page that does not contain the quoted passage, which is worse than no
# citation at all in an assignment about source grounding.
#
# THE BUG THIS REPLACES. The readers used to record offsets against the RAW extracted text,
# and `load_documents()` then collapsed all whitespace before storing it. Every offset after
# the first run of whitespace was wrong, and because PDF extraction is whitespace-heavy the
# error accumulated: in the WEF report, content on page 48 was cited as page 10.
#
# The fix is to normalise EACH PAGE first, then build both the document text and the offsets
# from those normalised pages, so the two can never disagree.
PAGE_OFFSETS: dict[str, list[int]] = {}

PAGE_JOIN = " "


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _assemble(name: str, pages: list[str]) -> str:
    """Join normalised pages and record where each one starts. The single source of truth."""
    offsets, parts, pos = [], [], 0
    for p in pages:
        offsets.append(pos)
        parts.append(p)
        pos += len(p) + len(PAGE_JOIN)
    PAGE_OFFSETS[name] = offsets
    return PAGE_JOIN.join(parts)


def page_of(name: str, char_index: int) -> int | None:
    """1-indexed page containing a character offset, or None for non-paginated sources."""
    offs = PAGE_OFFSETS.get(name)
    return bisect.bisect_right(offs, char_index) if offs else None


def cite(name: str, char_index: int) -> str:
    p = page_of(name, char_index)
    return f"{name} p.{p}" if p else name


def _read_pdf(path: Path) -> str:
    """Normalised text, with per-page offsets recorded against that same normalised text."""
    from pypdf import PdfReader
    try:
        reader = PdfReader(str(path))
        pages = [_normalise(pg.extract_text() or "") for pg in reader.pages]
        return _assemble(path.name, pages)
    except Exception as e:  # noqa: BLE001
        # The message, not just the class. An earlier version printed only the exception
        # type, and a NameError introduced by an edit read as "this PDF is unreadable" for
        # every file in the folder.
        print(f"    WARNING: could not read {path.name} - {type(e).__name__}: {e}")
        return ""


def _read_html(path: Path) -> str:
    """HTML has no pages, so no offsets are recorded and citations name the file only."""
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="replace"), "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer"]):
        tag.decompose()
    return _normalise(soup.get_text(" "))


def source_files(directory: Path | None = None) -> list[Path]:
    """
    THE one definition of "a Shock source document", used everywhere.

    It excludes the folder's own instructions, the source DECLARATION and any dotfiles.
    Those are committed to the repository; the sources are not, and are not
    redistributable. Keeping two different notions of the set broke the source-free
    checkout: `input_fingerprints()` had its own list that included README.md and
    sources.json, so the fingerprint dictionary never emptied when the documents were
    removed, its fallback never fired, and the recorded map could not match the live one
    in any case.

    Returns [] rather than raising, so a caller can ask "are the sources here?" without
    handling an exception.
    """
    directory = directory or CONTEXT_DIR
    directory.mkdir(parents=True, exist_ok=True)
    return [f for f in sorted(directory.iterdir())
            if f.is_file()
            and f.suffix.lower() in SUPPORTED
            and f.stem.lower() not in {"readme", "desktop", "sources"}
            and not f.name.startswith(".")]


def load_documents(directory: Path | None = None) -> dict[str, str]:
    """
    Read every source document in the context directory. Returns {filename: text}.

    Raises if there are none, because a silent empty result here would produce
    scenarios generated from the model's general knowledge - exactly what the Shock stage forbids -
    and nothing downstream would tell you.
    """
    directory = directory or CONTEXT_DIR
    files = source_files(directory)

    if not files:
        raise FileNotFoundError(
            f"No context documents found in {directory}.\n"
            f"The Shock stage requires you to download them yourself - the course cannot "
            f"redistribute them. See {directory / 'README.md'}.")

    out: dict[str, str] = {}
    for f in files:
        ext = f.suffix.lower()
        # The readers normalise per page and record offsets against that text. Do NOT
        # normalise again here - the previous version did, which silently invalidated
        # every page offset the readers had just computed.
        text = (_read_pdf(f) if ext == ".pdf"
                else _read_html(f) if ext in (".html", ".htm")
                else _normalise(f.read_text(encoding="utf-8", errors="replace")))
        if len(text) < 500:
            print(f"    WARNING: {f.name} yielded only {len(text)} characters - "
                  f"is it a scanned PDF with no text layer?")
            continue
        out[f.name] = text

    print(f"  loaded {len(out)} context documents from {directory}")
    for name, text in out.items():
        print(f"    {name:52s} {len(text):>8,} chars")
    if len(out) < 3:
        print(f"  WARNING: only {len(out)} usable documents. The Shock stage expects at least "
              f"three independent sources - one report's risk taxonomy is not a survey.")
    return out


def relevant_passages(docs: dict[str, str], query_terms: list[str],
                      budget_chars: int = DEFAULT_BUDGET_CHARS,
                      window: int = 1200,
                      return_tags: bool = False):
    # With return_tags=True this returns (text, {tag: passage}). The MAP matters, not just
    # the tag set: a citation can only be verified against the passage it names, and a
    # channel's quote has to be checkable against that text.
    """
    Keyword-windowed retrieval across the loaded documents.

    Deliberately simple and deliberately transparent: for each query term, take a window of
    characters around each hit, deduplicate, and pack up to the budget. You can see exactly
    what reached the prompt, which you cannot with an opaque embedding retriever - and for
    the Shock stage that auditability matters more than retrieval quality.

    Every passage is prefixed with its source filename so the model can cite it and you can
    check the citation.
    """
    # Collect hits PER DOCUMENT first, then interleave. Packing documents one after another
    # lets the longest source consume the whole budget before the others are reached - and
    # with reports of very different lengths that happens every time. Round-robin gives
    # every document you downloaded a share of the prompt.
    by_doc: dict[str, list[str]] = {}
    seen: set[str] = set()
    for name, text in docs.items():
        # Search the ORIGINAL text case-insensitively rather than a lowered copy.
        # `.lower()` is not length-preserving for every Unicode character, and the page
        # offsets were computed on the original - so one such character would shift
        # every citation after it in that document.
        hits: list[tuple[int, str]] = []
        for term in query_terms:
            for m in re.finditer(re.escape(term), text, re.IGNORECASE):
                a = max(0, m.start() - window // 2)
                b = min(len(text), m.start() + window // 2)
                snippet = text[a:b].strip()
                key = snippet[:120]
                if key in seen:
                    continue
                seen.add(key)
                hits.append((m.start(), snippet))
        if hits:
            by_doc[name] = hits

    picked: list[tuple[str, int, str]] = []
    for i in range(max((len(v) for v in by_doc.values()), default=0)):
        for name, hits in by_doc.items():
            if i < len(hits):
                picked.append((name, hits[i][0], hits[i][1]))

    if not picked:
        print(f"    WARNING: no passages matched {query_terms}.")
        print(f"    Either your documents do not cover this scenario, or your retrieval "
              f"terms are wrong. The model would be reasoning from the scenario "
              f"description alone, which the Shock stage does not allow.")
        return ("", {}) if return_tags else ""

    out, total, used, tags = [], 0, {}, {}
    for name, pos, snippet in picked:
        tag = cite(name, pos)
        block = f"[{tag}] ...{snippet}...\n"
        if total + len(block) > budget_chars:
            # Do NOT record the tag here. It used to be added before this
            # check, so a source could count as "retrieved" - and satisfy
            # require_sources() - when none of its text reached the model.
            break
        tags[tag] = snippet
        out.append(block)
        total += len(block)
        used[name] = used.get(name, 0) + 1

    # Per-source reporting. Teams download different documents, so the useful question is
    # not "did retrieval work" but "which of MY documents actually fed this scenario".
    print(f"    retrieved {len(out)} passages ({total:,} chars)")
    for name in docs:
        n = used.get(name, 0)
        flag = "   <-- contributed nothing" if n == 0 else ""
        print(f"      {name:50s} {n:3d} passages{flag}")
    if len(used) == 1 and len(docs) > 1:
        print(f"    WARNING: every passage came from one document. Your channels will "
              f"reflect that source's house view rather than a survey of the risk.")
    text = "\n".join(out)
    return (text, tags) if return_tags else text


def inventory() -> None:
    """`python src/context_docs.py` - check your downloads before spending API credit."""
    try:
        docs = load_documents()
    except FileNotFoundError as e:
        print(e)
        return
    print(f"\n  total {sum(len(t) for t in docs.values()):,} characters available")


if __name__ == "__main__":
    inventory()


# -------------------------------------------------------------------------------------------
# Source accounting
# -------------------------------------------------------------------------------------------

MIN_SOURCES = 3


def manifest(docs: dict[str, str], directory: Path | None = None) -> list[dict]:
    """
    One record per downloaded document: organisation, title, URL, size, pages and a hash.

    The organisation comes from `data/raw/context/sources.json` if you wrote one, and is
    otherwise guessed from the filename. Guessing is not identification - three files from
    one publisher can pass the three-organisation rule on their names alone, and one badly
    named file can fail it - so the manifest is how you state what you actually downloaded.
    """
    import hashlib
    declared = _declared_sources(directory)
    out = []
    for name, text in sorted(docs.items()):
        d = declared.get(name, {})
        out.append({
            "file": name,
            "organisation": d.get("organisation") or _guess_org(name),
            "organisation_declared": bool(d.get("organisation")),
            "title": d.get("title"),
            "url": d.get("url"),
            "retrieved": d.get("retrieved"),
            "chars": len(text),
            "pages": len(PAGE_OFFSETS.get(name, [])) or None,
            "sha256_16": hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()[:16],
        })
    return out


def _guess_org(name: str) -> str:
    return name.split("-")[0].lower() if "-" in name else name.split(".")[0].lower()


DECLARATION_FIELDS = ("file", "organisation", "title", "url", "retrieved")


def _norm_org(name: str) -> str:
    """
    Fold an organisation name for COUNTING distinct publishers.

    "IMF", "imf" and "I.M.F." are one organisation. The comparison used to be a bare set of
    the strings as typed, so case and punctuation could turn one publisher into three and
    pass the three-organisation rule on typography.
    """
    return "".join(ch for ch in (name or "").lower() if ch.isalnum())


def _declared_sources(directory: Path | None = None) -> dict[str, dict]:
    """
    Read and VALIDATE `sources.json`. A malformed declaration raises rather than degrading.

    The brief tells teams to record organisation, title, URL and retrieval date for every
    document. The check used to look at `organisation` only, so a record with an empty
    title, no URL and no date passed, and a file could be silently absent from the manifest
    because its record had no `file` key. Every field is now required, the date is parsed,
    and a bad record names itself.
    """
    directory = directory or CONTEXT_DIR
    f = directory / "sources.json"
    if not f.exists():
        return {}
    try:
        raw = json.loads(f.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            f"{f} is not valid JSON ({e}). Fix it or delete it - a source declaration that "
            f"cannot be read is worse than none, because the run would carry on with "
            f"organisations guessed from filenames.") from None
    if not isinstance(raw, list):
        raise RuntimeError(f"{f} must be a JSON list of records, one per document.")
    out, problems, combos = {}, [], {}
    for n, r in enumerate(raw, 1):
        if not isinstance(r, dict):
            problems.append(f"record {n} is not an object")
            continue
        who = r.get("file") or f"record {n}"
        gaps = [k for k in DECLARATION_FIELDS if not str(r.get(k, "")).strip()]
        if gaps:
            problems.append(f"{who} is missing {gaps}")
            continue
        try:
            got = datetime.strptime(str(r["retrieved"]).strip(), "%Y-%m-%d")
        except ValueError:
            problems.append(f"{who}: retrieved={r['retrieved']!r} is not YYYY-MM-DD")
            continue
        if got.date() > datetime.now().date():
            problems.append(f"{who}: retrieved={r['retrieved']!r} is in the future - a "
                            f"document cannot have been downloaded on a date that has "
                            f"not happened")
            continue
        if not str(r["url"]).strip().lower().startswith(("http://", "https://")):
            problems.append(f"{who}: url={r['url']!r} is not a URL")
            continue
        key = str(r["file"]).strip()
        if key in out:
            # a dict would keep only the LAST record for a duplicated filename, so an
            # accidental copy-paste silently replaced a declaration instead of failing
            problems.append(f"{who}: duplicate declaration for the same file")
            continue
        combo = (_norm_org(str(r["organisation"])),
                 " ".join(str(r["title"]).split()).lower())
        if combo in combos:
            problems.append(f"{who}: declares the same organisation and title as "
                            f"{combos[combo]!r} - one document declared twice under two "
                            f"filenames cannot both be real")
            continue
        combos[combo] = key
        out[key] = r
    if problems:
        raise RuntimeError(
            f"{len(problems)} problem(s) in {f}:" + chr(10) + "  - "
            + (chr(10) + "  - ").join(problems) + chr(10)
            + f"Every record needs {list(DECLARATION_FIELDS)}, a retrieval date as "
            + "YYYY-MM-DD and a real URL. See sources.json.template.")
    return out


def require_sources(man: list[dict], tags, scenario: str = "") -> None:
    """
    BLOCK if retrieval did not actually deliver evidence. Warnings were not enough.

    A scenario whose retrieval returned nothing still ran, and the model then produced
    channels from the scenario description and its own training data - which is precisely
    what the Shock stage forbids. Three failure modes now raise:

      - fewer than MIN_SOURCES documents in the folder
      - fewer than MIN_SOURCES distinct organisations among them
      - retrieval that returned passages from fewer than two documents
    """
    if len(man) < MIN_SOURCES:
        raise RuntimeError(
            f"{len(man)} context document(s) found; {MIN_SOURCES} are required. "
            f"See data/raw/context/README.md.")
    orgs = {_norm_org(m["organisation"]) for m in man}
    undeclared = [m["file"] for m in man if not m.get("organisation_declared")]
    declared_files = set(_declared_sources())
    orphaned = sorted(declared_files - {m["file"] for m in man})
    if orphaned:
        raise RuntimeError(
            f"sources.json declares {orphaned}, which are not in data/raw/context/. Either "
            f"the download is missing or the declaration names the wrong file - and a "
            f"declaration that does not match the folder cannot be checked by a marker.")
    if undeclared:
        raise RuntimeError(
            f"{len(undeclared)} source(s) have no entry in data/raw/context/sources.json: "
            f"{undeclared}. Copy sources.json.template and fill in "
            f"organisation, title, url and retrieved for each file. Guessing the "
            f"organisation from the filename is not identification - three reports "
            f"from one publisher would pass the three-organisation rule on their "
            f"names alone.")
    if len(orgs) < MIN_SOURCES:
        raise RuntimeError(
            f"{len(man)} files but only {len(orgs)} distinct organisations ({sorted(orgs)}). "
            f"One publisher's house view is not a survey of the risk - download from "
            f"{MIN_SOURCES} different organisations.")
    files_hit = {t.split(" p.")[0] for t in tags}
    if not tags:
        raise RuntimeError(
            f"retrieval returned NO passages for {scenario or 'this scenario'}. The model "
            f"would be reasoning from the scenario description and its own training data, "
            f"which this stage does not allow. Fix your retrieval terms or download "
            f"documents that cover it.")
    if len(files_hit) < 2:
        raise RuntimeError(
            f"retrieval for {scenario or 'this scenario'} matched only {sorted(files_hit)}. "
            f"A single source's framing is not grounding - broaden your retrieval terms.")
    print(f"    sources: {len(man)} documents, {len(orgs)} organisations, "
          f"{len(tags)} passage tags across {len(files_hit)} files")
