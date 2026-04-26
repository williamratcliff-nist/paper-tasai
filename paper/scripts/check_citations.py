#!/usr/bin/env python3
"""
check_citations.py — read-only sanity check for references.bib against Crossref
and against the manuscript's actual \\cite usage.

What it does, in three passes:

  1. Crossref diff. For each bib entry, query the Crossref REST API and flag
     mismatches in title, year, DOI, and the full author list (both family
     and given names, position by position).
  2. Manuscript scan. Scan the two markdown sources for pandoc-style [@key]
     citations, then report (a) cited keys that have no bib entry, and
     (b) bib entries that are never cited.
  3. Report. Emit a grouped markdown report to stdout or a file.

Stdlib only — no pip install required. The manuscript is not modified.
Crossref matches are nearest-by-title; treat the output as a prompt for
human review, not an oracle.

Usage:
    python paper/scripts/check_citations.py
    python paper/scripts/check_citations.py --bib path/to/references.bib
    python paper/scripts/check_citations.py --out citation_audit.md
    python paper/scripts/check_citations.py --only keimer2015,burger2020
    python paper/scripts/check_citations.py --no-crossref   # just the scan
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

CROSSREF_URL = "https://api.crossref.org/works"
USER_AGENT = "tasai-bib-check/0.2 (mailto:wratclif@umd.edu)"
PAPER_DIR = Path(__file__).resolve().parent.parent
DEFAULT_BIB = PAPER_DIR / "references.bib"
DEFAULT_MANUSCRIPTS = [
    PAPER_DIR / "digital_discovery_paper.md",
    PAPER_DIR / "TAS-AI_Digital_Discovery_SI.md",
]
CITE_RE = re.compile(r"\[@([A-Za-z][A-Za-z0-9_:\-]*)")


# ---------- minimal bibtex parser ----------

ENTRY_RE = re.compile(r"@(\w+)\s*\{\s*([^,\s]+)\s*,", re.MULTILINE)


@dataclass
class BibEntry:
    key: str
    kind: str
    fields: dict = field(default_factory=dict)

    def get(self, name: str, default: str = "") -> str:
        return self.fields.get(name.lower(), default)

    @property
    def first_author_surname(self) -> str:
        parsed = self.authors_parsed
        return parsed[0][1] if parsed else ""

    @property
    def authors_parsed(self) -> list[tuple[str, str]]:
        raw = self.get("author")
        if not raw:
            return []
        return [_parse_author_name(a) for a in raw.split(" and ") if a.strip() and a.strip().lower() != "others"]


def parse_bib(text: str) -> list[BibEntry]:
    entries = []
    for m in ENTRY_RE.finditer(text):
        kind = m.group(1).lower()
        key = m.group(2)
        start = m.end()
        depth = 1
        i = start
        while i < len(text) and depth > 0:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        body = text[start : i - 1]
        fields = _parse_fields(body)
        entries.append(BibEntry(key=key, kind=kind, fields=fields))
    return entries


def _parse_fields(body: str) -> dict:
    fields = {}
    i = 0
    while i < len(body):
        m = re.match(r"\s*(\w+)\s*=\s*", body[i:])
        if not m:
            break
        name = m.group(1).lower()
        i += m.end()
        if i >= len(body):
            break
        ch = body[i]
        if ch == "{":
            depth, j = 1, i + 1
            while j < len(body) and depth > 0:
                if body[j] == "{":
                    depth += 1
                elif body[j] == "}":
                    depth -= 1
                j += 1
            value = body[i + 1 : j - 1]
            i = j
        elif ch == '"':
            j = i + 1
            while j < len(body) and body[j] != '"':
                j += 1
            value = body[i + 1 : j]
            i = j + 1
        else:
            m2 = re.match(r"([^,\s]+)", body[i:])
            value = m2.group(1) if m2 else ""
            i += len(value)
        fields[name] = _clean(value)
        m3 = re.match(r"\s*,\s*", body[i:])
        i += m3.end() if m3 else 0
    return fields


def _clean(s: str) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    s = s.replace("{", "").replace("}", "")
    s = s.replace("\\&", "&")
    return s


def _parse_author_name(s: str) -> tuple[str, str]:
    """Return (given, family) for a single bibtex author token.

    Handles both "Family, Given M." and "Given M. Family" conventions.
    Initials and middle names are preserved in the given string.
    """
    s = s.strip()
    if not s:
        return ("", "")
    if "," in s:
        family, given = s.split(",", 1)
        return (given.strip(), family.strip())
    parts = s.split()
    if len(parts) == 1:
        return ("", parts[0])
    return (" ".join(parts[:-1]), parts[-1])


# Common LaTeX accent escapes \u2192 Unicode (covers the cases that show up in our bib).
_LATEX_ACCENTS = {
    r"\'a": "\u00e1", r"\'e": "\u00e9", r"\'i": "\u00ed", r"\'o": "\u00f3", r"\'u": "\u00fa", r"\'y": "\u00fd",
    r"\'A": "\u00c1", r"\'E": "\u00c9", r"\'I": "\u00cd", r"\'O": "\u00d3", r"\'U": "\u00da",
    r"\`a": "\u00e0", r"\`e": "\u00e8", r"\`i": "\u00ec", r"\`o": "\u00f2", r"\`u": "\u00f9",
    r'\"a': "\u00e4", r'\"e': "\u00eb", r'\"i': "\u00ef", r'\"o': "\u00f6", r'\"u': "\u00fc",
    r'\"A': "\u00c4", r'\"O': "\u00d6", r'\"U': "\u00dc",
    r"\^a": "\u00e2", r"\^e": "\u00ea", r"\^i": "\u00ee", r"\^o": "\u00f4", r"\^u": "\u00fb",
    r"\~n": "\u00f1", r"\~N": "\u00d1", r"\~a": "\u00e3", r"\~o": "\u00f5",
    r"\c c": "\u00e7", r"\c C": "\u00c7",
    r"\ss": "\u00df", r"\o": "\u00f8", r"\O": "\u00d8", r"\aa": "\u00e5", r"\AA": "\u00c5",
}


def _delatex(s: str) -> str:
    """Best-effort: convert common LaTeX accent macros to their Unicode form."""
    for k, v in _LATEX_ACCENTS.items():
        s = s.replace(k, v)
        # also handle the {\'a} / {\"o} brace form
        s = s.replace("{" + k + "}", v)
    return s


def _name_token(s: str) -> str:
    """Canonicalize a name token for comparison: convert LaTeX accents to
    Unicode, lowercase, strip punctuation, collapse whitespace."""
    s = _delatex(s)
    s = s.lower()
    s = re.sub(r"[\.\-']", " ", s)
    s = re.sub(r"[^a-z\u00c0-\u024f\s]", "", s)
    return re.sub(r"\s+", " ", s).strip()


def _family_match(bib_family: str, xref_family: str) -> bool:
    """Permissive family-name compare. Handles multi-word surnames where one
    side has a particle (e.g. 'Le Goc' vs 'Goc' from Crossref's broken split,
    'van Dover' vs 'Dover')."""
    a = _name_token(bib_family)
    b = _name_token(xref_family)
    if a == b:
        return True
    # last-token equivalence: 'le goc' ~ 'goc'
    a_last = a.split()[-1] if a else ""
    b_last = b.split()[-1] if b else ""
    if a_last and a_last == b_last:
        return True
    # one side fully contained in the other (handles particles like 'van', 'de')
    if a and b and (a.endswith(b) or b.endswith(a)):
        return True
    return False


def _given_match(bib_given: str, xref_given: str) -> bool:
    """Match given names permissively: "Chi" vs "Chiheb" should NOT match,
    but "C." vs "Chi" or "Chi R." vs "Chi" should. Compare first token only,
    allowing either side to be an initial of the other."""
    a = _name_token(bib_given).split()
    b = _name_token(xref_given).split()
    if not a or not b:
        return True  # can't compare, don't flag
    a0, b0 = a[0], b[0]
    if a0 == b0:
        return True
    # initial match: one side is a 1-letter initial of the other
    if len(a0) == 1 and b0.startswith(a0):
        return True
    if len(b0) == 1 and a0.startswith(b0):
        return True
    return False


# ---------- Crossref query ----------


def crossref_lookup(entry: BibEntry) -> Optional[dict]:
    """Fetch the canonical Crossref record for a bib entry.

    If the entry has a `doi` field, look it up directly via the works/{doi}
    endpoint — this is authoritative and avoids the fuzzy-search wrong-record
    problem (e.g. matching a journal version when the bib cites the proceedings
    version). Falls back to title+author bibliographic search only when no DOI
    is present.
    """
    doi = entry.get("doi").strip()
    if doi:
        # strip any URL prefix users sometimes paste in
        doi = re.sub(r"^https?://(dx\.)?doi\.org/", "", doi, flags=re.I)
        url = f"{CROSSREF_URL}/{urllib.parse.quote(doi, safe='/')}"
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            return {"_error": f"DOI lookup failed for {doi}: {e}"}
        msg = data.get("message")
        if msg:
            return msg
        return None

    title = entry.get("title")
    if not title:
        return None
    params = {"query.bibliographic": title, "rows": "3"}
    author = entry.first_author_surname
    if author:
        params["query.author"] = author
    url = f"{CROSSREF_URL}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        return {"_error": str(e)}
    items = data.get("message", {}).get("items", [])
    if not items:
        return None
    # pick best by fuzzy title match
    bib_title_norm = _norm(title)
    best = max(
        items,
        key=lambda it: _title_similarity(bib_title_norm, _norm(" ".join(it.get("title", [""])))),
    )
    return best


_MATHML_RE = re.compile(r"<mml:[^>]*>|</mml:[^>]*>", re.IGNORECASE)


def _strip_mathml(s: str) -> str:
    """Remove MathML tags Crossref sometimes leaves in titles."""
    return _MATHML_RE.sub(" ", s)


def _norm(s: str) -> str:
    s = _strip_mathml(s)
    s = _delatex(s)
    # Replace non-alphanumeric (e.g. LaTeX $_2$, $^4$) with space so chemical
    # formulas in bib like "La$_2$CuO$_4$" tokenize the same way as the MathML
    # form Crossref returns ("La 2 CuO 4").
    s = re.sub(r"[^a-z0-9 ]+", " ", s.lower())
    return re.sub(r"\s+", " ", s).strip()


def _title_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    aw, bw = set(a.split()), set(b.split())
    return len(aw & bw) / max(1, len(aw | bw))


# ---------- diff ----------


@dataclass
class Finding:
    key: str
    field: str
    bib: str
    crossref: str
    severity: str = "warn"  # "warn" or "info"


def diff_entry(entry: BibEntry, xref: dict) -> list[Finding]:
    findings: list[Finding] = []
    if not xref:
        findings.append(Finding(entry.key, "crossref", entry.get("title"), "(no match)", "warn"))
        return findings
    if xref.get("_error"):
        findings.append(Finding(entry.key, "crossref", "", xref["_error"], "info"))
        return findings

    # title
    bib_title = _norm(entry.get("title"))
    xr_title = _norm(" ".join(xref.get("title", [""])))
    sim = _title_similarity(bib_title, xr_title)

    # If the bib has no DOI and the fuzzy-matched record looks like a different
    # paper (low title similarity AND year drift), emit a single info-level
    # finding and skip the per-field diff: doing it would produce a wall of
    # noise comparing against the wrong record. This is the common case for
    # pre-DOI conference proceedings (e.g. ICML 2010).
    bib_year = entry.get("year")
    xr_year = ""
    for k in ("published-print", "published-online", "issued"):
        if k in xref:
            parts = xref[k].get("date-parts", [[None]])
            xr_year = str(parts[0][0]) if parts and parts[0] else ""
            if xr_year:
                break
    if not entry.get("doi"):
        try:
            year_drift = abs(int(bib_year) - int(xr_year)) if (bib_year and xr_year) else 0
        except ValueError:
            year_drift = 0
        if sim < 0.6 and year_drift >= 2:
            findings.append(Finding(
                entry.key, "no_doi_uncertain_match",
                f"{entry.get('title')} ({bib_year})",
                f"{' '.join(xref.get('title', ['']))} ({xr_year}) — nearest Crossref match; likely a different record",
                "info",
            ))
            return findings

    if sim < 0.6:
        findings.append(
            Finding(entry.key, "title", entry.get("title"), " ".join(xref.get("title", [""])), "warn")
        )

    # full author list, position by position
    bib_authors = entry.authors_parsed
    xr_authors = [(a.get("given", ""), a.get("family", "")) for a in xref.get("author", [])]

    if bib_authors and xr_authors:
        n = min(len(bib_authors), len(xr_authors))
        for i in range(n):
            bib_given, bib_family = bib_authors[i]
            xr_given, xr_family = xr_authors[i]
            tag = "first_author" if i == 0 else f"author[{i}]"
            if not _family_match(bib_family, xr_family):
                findings.append(
                    Finding(entry.key, f"{tag}.family", f"{bib_given} {bib_family}".strip(),
                            f"{xr_given} {xr_family}".strip(), "warn")
                )
            elif not _given_match(bib_given, xr_given):
                findings.append(
                    Finding(entry.key, f"{tag}.given", bib_given or "(none)",
                            xr_given or "(none)", "warn")
                )
        if len(bib_authors) < len(xr_authors):
            missing = xr_authors[len(bib_authors):]
            findings.append(
                Finding(entry.key, "author_missing",
                        f"bib stops at {len(bib_authors)}",
                        "; ".join(f"{g} {f}".strip() for g, f in missing[:4]) + (" ..." if len(missing) > 4 else ""),
                        "info")
            )
        elif len(bib_authors) > len(xr_authors):
            extra = bib_authors[len(xr_authors):]
            findings.append(
                Finding(entry.key, "author_extra",
                        "; ".join(f"{g} {f}".strip() for g, f in extra[:4]) + (" ..." if len(extra) > 4 else ""),
                        f"crossref stops at {len(xr_authors)}",
                        "warn")
            )

    # year
    bib_year = entry.get("year")
    xr_year = ""
    for key in ("published-print", "published-online", "issued"):
        if key in xref:
            parts = xref[key].get("date-parts", [[None]])
            xr_year = str(parts[0][0]) if parts and parts[0] else ""
            if xr_year:
                break
    if bib_year and xr_year and bib_year != xr_year:
        findings.append(Finding(entry.key, "year", bib_year, xr_year, "warn"))

    # DOI presence
    xr_doi = xref.get("DOI", "")
    bib_doi = entry.get("doi")
    if xr_doi and not bib_doi:
        findings.append(Finding(entry.key, "doi_missing", "(none)", xr_doi, "info"))

    return findings


# ---------- manuscript scan ----------


def scan_citations(md_paths: list[Path]) -> dict[str, list[tuple[Path, int]]]:
    """Return {cite_key: [(path, line), ...]} for every pandoc-style [@key]
    citation in the given markdown files. Handles multiple keys per bracket."""
    uses: dict[str, list[tuple[Path, int]]] = {}
    for p in md_paths:
        if not p.exists():
            continue
        for lineno, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            for m in CITE_RE.finditer(line):
                key = m.group(1)
                uses.setdefault(key, []).append((p, lineno))
                # also catch [@k1; @k2; ...] — keep scanning the remainder
            for m in re.finditer(r";\s*@([A-Za-z][A-Za-z0-9_:\-]*)", line):
                key = m.group(1)
                uses.setdefault(key, []).append((p, lineno))
    return uses


# ---------- report ----------


def render_report(findings_by_key: dict, entries: list[BibEntry],
                   cite_uses: Optional[dict] = None,
                   skipped_crossref: bool = False) -> str:
    lines = ["# Citation audit report", ""]
    lines.append(f"- Bib entries parsed: **{len(entries)}**")
    if skipped_crossref:
        lines.append("- Crossref diff: **skipped** (`--no-crossref`)")
    else:
        n_keys_flagged = sum(1 for fs in findings_by_key.values() if any(f.severity == "warn" for f in fs))
        lines.append(f"- Entries with Crossref warnings: **{n_keys_flagged}**")
    if cite_uses is not None:
        bib_keys = {e.key for e in entries}
        unresolved = sorted(set(cite_uses) - bib_keys)
        unused = sorted(bib_keys - set(cite_uses))
        lines.append(f"- Cite keys in manuscript: **{len(cite_uses)}** unique")
        lines.append(f"- Unresolved cite keys (cited but not in bib): **{len(unresolved)}**")
        lines.append(f"- Unused bib entries (in bib but never cited): **{len(unused)}**")
    lines.append("")
    lines.append("Source: Crossref REST API. Matches are nearest-by-title; human review required.")
    lines.append("")

    # warnings first
    lines.append("## Warnings (likely errors)")
    any_warn = False
    for entry in entries:
        fs = [f for f in findings_by_key.get(entry.key, []) if f.severity == "warn"]
        if not fs:
            continue
        any_warn = True
        lines.append(f"### `{entry.key}`")
        for f in fs:
            lines.append(f"- **{f.field}**")
            lines.append(f"  - bib: `{f.bib}`")
            lines.append(f"  - crossref: `{f.crossref}`")
        lines.append("")
    if not any_warn:
        lines.append("_None._")
        lines.append("")

    # info
    lines.append("## Informational")
    any_info = False
    for entry in entries:
        fs = [f for f in findings_by_key.get(entry.key, []) if f.severity == "info"]
        if not fs:
            continue
        any_info = True
        lines.append(f"### `{entry.key}`")
        for f in fs:
            lines.append(f"- **{f.field}**: bib `{f.bib}` → crossref `{f.crossref}`")
        lines.append("")
    if not any_info:
        lines.append("_None._")
    lines.append("")

    # citation-key resolution
    if cite_uses is not None:
        bib_keys = {e.key for e in entries}
        unresolved = sorted(set(cite_uses) - bib_keys)
        unused = sorted(bib_keys - set(cite_uses))

        lines.append("## Manuscript citation resolution")
        lines.append("")
        lines.append("### Unresolved cite keys (cited in manuscript, missing from bib)")
        if unresolved:
            for k in unresolved:
                locs = cite_uses[k]
                loc_str = "; ".join(f"{p.name}:{ln}" for p, ln in locs[:3])
                if len(locs) > 3:
                    loc_str += f" (+{len(locs) - 3} more)"
                lines.append(f"- `{k}` — {loc_str}")
        else:
            lines.append("_None._")
        lines.append("")

        lines.append("### Unused bib entries (in bib, never cited)")
        if unused:
            for k in unused:
                lines.append(f"- `{k}`")
        else:
            lines.append("_None._")
        lines.append("")

    return "\n".join(lines)


# ---------- main ----------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bib", type=Path, default=DEFAULT_BIB)
    ap.add_argument("--out", type=Path, default=None, help="write markdown report to this path")
    ap.add_argument("--delay", type=float, default=0.2, help="seconds between Crossref requests")
    ap.add_argument("--only", type=str, default=None, help="comma-separated bib keys to check")
    ap.add_argument("--no-crossref", action="store_true", help="skip Crossref diff (only scan manuscript for unresolved/unused keys)")
    ap.add_argument("--md", type=Path, nargs="*", default=None,
                    help="markdown files to scan for [@key] usage (default: paper/*.md)")
    args = ap.parse_args()

    if not args.bib.exists():
        print(f"bib file not found: {args.bib}", file=sys.stderr)
        return 2

    entries = parse_bib(args.bib.read_text(encoding="utf-8"))
    if args.only:
        wanted = {k.strip() for k in args.only.split(",")}
        crossref_entries = [e for e in entries if e.key in wanted]
    else:
        crossref_entries = entries

    findings_by_key: dict = {}
    if not args.no_crossref:
        print(f"Checking {len(crossref_entries)} entries against Crossref...", file=sys.stderr)
        for i, entry in enumerate(crossref_entries, 1):
            print(f"  [{i:>2}/{len(crossref_entries)}] {entry.key}", file=sys.stderr)
            xref = crossref_lookup(entry)
            findings_by_key[entry.key] = diff_entry(entry, xref)
            time.sleep(args.delay)

    md_paths = args.md if args.md is not None else DEFAULT_MANUSCRIPTS
    cite_uses = scan_citations([Path(p) for p in md_paths])
    print(f"Scanned {len(md_paths)} manuscript file(s); {len(cite_uses)} unique cite keys found.", file=sys.stderr)

    report = render_report(findings_by_key, entries, cite_uses, skipped_crossref=args.no_crossref)
    if args.out:
        args.out.write_text(report, encoding="utf-8")
        print(f"Report written to {args.out}", file=sys.stderr)
    else:
        print(report)

    n_warn = sum(1 for fs in findings_by_key.values() if any(f.severity == "warn" for f in fs))
    bib_keys = {e.key for e in entries}
    n_unresolved = len(set(cite_uses) - bib_keys)
    return 1 if (n_warn or n_unresolved) else 0


if __name__ == "__main__":
    sys.exit(main())
