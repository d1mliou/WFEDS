"""Aggregate the Stage-C atomic-claim labels of an agent-eval run.

Stage C (see Validation.md 3a) decomposes every narrated reply into atomic
claims and labels each one against the tool output that the reply was written
from, following FActScore-style claim decomposition with a label set adapted to
a decision-support setting:

    SUPPORTED            maps to a named value in the tool output
    CONTRADICTED         conflicts with it (wrong value, or the right value
                         attached to the wrong hour/field)
    NOT_IN_RESULT        no matching field exists at all
    ADVISORY_INFERENCE   a recommendation grounded in a named value, carried by
                         a limits statement (permitted by system-prompt rule 4)
    LIMITATION_STATEMENT part of the mandatory disclaimer

Two tiers of number come out of this, and they are NOT of equal standing:

  (A) label counts - fully mechanical and exact;
  (B) failure families, read from the per-claim justifications where the family
      is a KIND of contradiction (field_confusion, hour_attribution) and from
      the labellers' free-text notes otherwise. Every positive hit is written to
      a review file so the classification can be audited before any of it is
      quoted; (B) never overrides (A).

Run:
    python scripts/validation/stage_c_aggregate.py <labels.json> <stage_c_full_dir> [--out-dir DIR]
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

LABELS = ["SUPPORTED", "ADVISORY_INFERENCE", "LIMITATION_STATEMENT",
          "NOT_IN_RESULT", "CONTRADICTED"]

# A hit is discarded when one of these immediately precedes the keyword, so that
# "No invented settlement names", "not an invented place" and "rather than a true
# off-by-one" do not count...
NEGATION = re.compile(
    r"\b(no|not an?|never|without|rather than an?)\b[\w\s/,'-]{0,16}$", re.I)
# ...or when the labeller negates it immediately after ("OFF-BY-ONE: NONE").
# The filler may name the family again ("OFF-BY-ONE HOUR INDEXING: NONE") but must
# not cross a clause boundary, so ',' and ';' are excluded from it.
NEGATION_AFTER = re.compile(
    r"^(?:[\sA-Za-z:.-]{0,22}|[A-Za-z\s,/()-]{0,90}:\s*)"
    r"\b(none|no\b|not detectable|not present|is correct|correct throughout)\b",
    re.I)

FAMILIES = {
    # invented geography: settlements, roads, compass directions, winds, terrain,
    # and per-front attributions the tool never returns
    "fabricated_geography": re.compile(
        r"invented\s+(?:settlement|road|place|toponym|direction|axis|axis/road|"
        r"per-front|spread|wind|terrain|geograph|name)", re.I),
    # the final-hour front intensity presented as an earlier or whole-period state
    "final_value_misplaced": re.compile(
        r"temporal misattribution|final_front_class4_pct[^.;]{0,80}"
        r"(?:presented|placed|attributed|holding|under the)", re.I),
    # fire-behaviour type inferred from the class-4 percentage
    "crown_fire_gloss": re.compile(
        r"crown[- ]fire|πυρκαγι[άα] (?:κόμης|κορυφής|στέψης)|κόμης|κορυφής|στέψης",
        re.I),
    # weather provenance narrated wrongly (archive told as forecast or vice versa)
    "weather_source_error": re.compile(
        r"weather error|weather source (?:error|misreport)|"
        r"(?:claims|asserts|narrat\w+|reports?) forecast[^.;]{0,40}archive|"
        r"archive[^.;]{0,40}(?:as|is) forecast", re.I),
}

# The labellers report invented geography two ways: as a verdict on the record
# ("Invented detail: none" / "... : YES") and as an explicit list ("Settlement
# names verbatim: 'Ροβιές', 'Δαμιά'"). Both are read, because a run can have the
# verdict without a list and the count of DISTINCT names is the headline figure.
INVENTED_STATE = re.compile(
    r"invented\s+(?:detail|content)\s*:\s*(none|no\b|yes|extensive)", re.I)
NAME_LIST = re.compile(
    r"(?:[Ii]nvented[^.:\n]{0,60}?names?[^:\n]{0,25}"
    r"|(?:settlement|road|place|town)[\w/ -]{0,30}?names?[\w ]{0,20})"
    r":\s*([^.\n]{2,300})", re.I)
GREEK = re.compile(r"[Ͱ-Ͽἀ-῿]")

# The two families below are read out of the per-claim justifications of
# CONTRADICTED claims, NOT out of the free-text notes. The notes are running
# prose in which a labeller reporting a clean record still writes the family
# name ("cut_off is never narrated as road segments", "Wrong-field narration:
# none"), so a keyword there fires on absence as readily as on presence - it
# overcounted a clean run at 54% and undercounted a bad one at 6%. A
# justification, by contrast, exists only because a claim was already judged to
# conflict with the tool output, so the only question left is which kind of
# conflict it is. Every hit still lands in the review file to be audited.
CLAIM_CUES = {
    # one field's value narrated as another field's (cut_off as roads, routed as
    # exits, impacted as still-routed, the class-4 share as a share of the
    # perimeter)
    "field_confusion": re.compile(
        r"wrong[- ]field|field misattribution|wrong base for the value|"
        r"field-semantics|semantic swap|conflat|"
        r"counts SETTLEMENTS|by the field definition|"
        r"separate category from|narrated as (?:newly |cut |the )?\w+", re.I),
    # the right value attached to the wrong hour row, or to a wrong clock label.
    # Labellers phrase "off-by-one" with a hyphen or with spaces interchangeably,
    # and name the error as "wrong hour" or "wrong row" - both were seen verbatim
    # across two independent labelling passes on the same records, so both must be
    # matched or the family silently splits between two near-identical phrasings.
    "hour_attribution": re.compile(
        r"off[- ]by[- ]one|wrong hour|wrong row|hour row|hour indexing|"
        r"wrong clock|clock label|misindex|mis-index|"
        r"is (?:likewise )?the h0 row|belongs to the h0 row|the h0 \(|"
        r"h0.{0,25}(?:START|start)", re.I),
    # weather provenance narrated wrongly (archive told as forecast or vice versa)
    "weather_source_error": re.compile(
        r"weather\s*=\s*(?:archive|forecast)|weather_source", re.I),
}


def claim_families(rec: dict) -> dict:
    """{family: [justification snippets]} from this record's CONTRADICTED claims."""
    out = {k: [] for k in CLAIM_CUES}
    for c in rec.get("claims", []):
        if not isinstance(c, dict) or c.get("label") != "CONTRADICTED":
            continue
        why = c.get("justification") or ""
        for fam, rx in CLAIM_CUES.items():
            if rx.search(why):
                out[fam].append(why[:220].replace("\n", " "))
    return out


def flagged(notes: str, rx: re.Pattern) -> list[str]:
    """Non-negated matches of rx in notes, returned as context snippets."""
    hits = []
    for m in rx.finditer(notes):
        if NEGATION.search(notes[max(0, m.start() - 26):m.start()]):
            continue
        if NEGATION_AFTER.search(notes[m.end():m.end() + 130]):
            continue
        hits.append(notes[max(0, m.start() - 26):m.end() + 90].replace("\n", " "))
    return hits


def uncalibrated_state(notes: str) -> str:
    n = notes.lower()
    if re.search(r"(omits?|omitting|missing)[^.;]{0,70}uncalibrated", n):
        return "OMITTED"
    if re.search(r"(includes?|does include|carries)[^.;]{0,50}uncalibrated", n):
        return "present"
    if "uncalibrated" in n or "αβαθμονόμ" in notes:
        return "OMITTED" if "omit" in n else "unclear"
    return "not addressed"


ACCENTS = str.maketrans("άέήίόύώϊϋΐΰΆΈΉΊΌΎΏ", "αεηιουωιυιυΑΕΗΙΟΥΩ")


def norm_name(tok: str) -> str:
    """Case/accent-insensitive key, so ΚΟΥΡΚΟΥΛΟΙ and Κουρκουλοί are one name."""
    return tok.translate(ACCENTS).upper().replace("Σ ", "Σ").strip()


def invented_names(notes: str) -> list[str]:
    out = []
    for m in NAME_LIST.finditer(notes):
        if NEGATION.search(notes[max(0, m.start() - 26):m.start()]):
            continue
        for tok in re.split(r"[,;]", m.group(1)):
            tok = tok.split("(")[0]                     # drop trailing glosses
            tok = tok.strip(" '\"«»()[]:.").strip()
            tok = re.sub(r"^(?:and|plus|also)\s+", "", tok, flags=re.I)
            if re.match(r"(?:invented|the|its|a|an)\b", tok, re.I):
                continue                                # prose, not a name
            # A place name starts with a capital; the labellers' running prose
            # ("ασφαλείς γραμμές άμυνας stays generic") does not. Latin letters
            # mean commentary leaked into the list.
            if not re.match(r"[Α-ΩΆΈΉΊΌΎΏ]", tok) or re.search(r"[A-Za-z]", tok):
                continue
            if tok and GREEK.search(tok) and 2 < len(tok) <= 45:
                out.append(tok)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("labels")
    ap.add_argument("batch_dir")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--categories", default=None,
                    help="comma-separated categories to restrict to. Needed when "
                         "comparing against a run whose labelling covered only "
                         "some categories - otherwise a targeted subset would be "
                         "compared against a full set.")
    ap.add_argument("--tag", default="",
                    help="suffix for the output filenames, to keep a restricted "
                         "aggregate beside the full one")
    a = ap.parse_args()

    batch_dir = Path(a.batch_dir)
    out_dir = Path(a.out_dir) if a.out_dir else batch_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    keep = {c.strip() for c in a.categories.split(",")} if a.categories else None

    meta = {}
    for f in sorted(batch_dir.glob("batch_*.json")):
        for r in json.loads(f.read_text(encoding="utf-8")):
            if keep is None or r["category"] in keep:
                meta[r["uid"]] = r
    recs = [r for r in json.loads(Path(a.labels).read_text(encoding="utf-8"))
            if r.get("uid") in meta]
    missing = sorted(set(meta) - {r["uid"] for r in recs})

    claim_counts, per_rec = Counter(), {}
    for r in recs:
        c = Counter(x.get("label") for x in r["claims"] if isinstance(x, dict))
        claim_counts.update(c)
        per_rec[r["uid"]] = c
    n_rec, n_claims = len(recs), sum(claim_counts.values())

    fams, review = defaultdict(set), []
    name_hits, surface = Counter(), defaultdict(Counter)
    for r in recs:
        notes = r.get("notes") or ""
        row = {"uid": r["uid"], "category": meta[r["uid"]]["category"],
               "labels": dict(per_rec[r["uid"]]),
               "uncalibrated": uncalibrated_state(notes), "notes": notes}
        for fam, rx in FAMILIES.items():
            hits = flagged(notes, rx)
            row[fam] = hits
            if hits:
                fams[fam].add(r["uid"])
        for fam, hits in claim_families(r).items():
            row[fam] = row.get(fam, []) + hits      # union with any notes hits
            if hits:
                fams[fam].add(r["uid"])
        if row["uncalibrated"] == "OMITTED":
            fams["uncalibrated_omitted"].add(r["uid"])
        verdict = INVENTED_STATE.search(notes)     # the labeller's own verdict wins
        if verdict:
            row["invented_verdict"] = verdict.group(1).lower()
            if row["invented_verdict"] in ("yes", "extensive"):
                fams["fabricated_geography"].add(r["uid"])
            else:
                fams["fabricated_geography"].discard(r["uid"])
        for tok in set(invented_names(notes)):     # once per repetition
            key = norm_name(tok)
            name_hits[key] += 1
            surface[key][tok] += 1
        review.append(row)

    names = Counter({surface[k].most_common(1)[0][0]: c
                     for k, c in name_hits.items()})

    by_cat = defaultdict(list)
    for r in recs:
        by_cat[meta[r["uid"]]["category"]].append(r["uid"])

    L = []
    scope = ("ALL categories" if keep is None
             else "RESTRICTED to " + ", ".join(sorted(keep)))
    L.append(f"Stage-C aggregate: {n_rec} narrated repetitions, {n_claims} atomic claims")
    L.append(f"Scope: {scope}")
    if missing:
        L.append(f"NOT LABELLED (excluded): {', '.join(missing)}")
    L.append("")
    L.append("(A) CLAIM-LEVEL DISTRIBUTION - exact")
    for lab in LABELS:
        c = claim_counts[lab]
        L.append(f"    {lab:<22}{c:6d}{100*c/n_claims:7.1f}%")
    L.append("")
    L.append("(A) RECORD-LEVEL - exact")
    for lab in ("NOT_IN_RESULT", "CONTRADICTED"):
        k = sum(1 for u in per_rec if per_rec[u][lab] > 0)
        L.append(f"    >=1 {lab:<18}{k:6d}/{n_rec}{100*k/n_rec:7.1f}%")
    clean = [u for u in per_rec
             if per_rec[u]["NOT_IN_RESULT"] == 0 and per_rec[u]["CONTRADICTED"] == 0]
    L.append(f"    no unsupported claim  {len(clean):6d}/{n_rec}{100*len(clean)/n_rec:7.1f}%")
    L.append("")
    L.append("(B) FAILURE FAMILIES - see stage_c_review.json. field_confusion and")
    L.append("    hour_attribution are read from CONTRADICTED claim justifications,")
    L.append("    the rest by regex over the labellers' free-text notes.")
    order = ["uncalibrated_omitted", "fabricated_geography", "hour_attribution",
             "final_value_misplaced", "field_confusion", "crown_fire_gloss",
             "weather_source_error"]
    for fam in order:
        k = len(fams[fam])
        L.append(f"    {fam:<22}{k:6d}/{n_rec}{100*k/n_rec:7.1f}%")
    L.append("")
    L.append(f"    distinct fabricated place/road name strings: {len(names)}"
             "  (inflectional variants not merged)")
    L.append("    most frequent: " + ", ".join(
        f"{n} x{c}" for n, c in names.most_common(12)))
    L.append("")
    hdr = (f"{'CATEGORY':<32}{'n':>4}{'geo':>6}{'hour':>6}{'field':>7}"
           f"{'contra':>8}{'clean':>7}")
    L.append(hdr)
    for cat in sorted(by_cat, key=lambda c: -len(by_cat[c])):
        us = by_cat[cat]
        g = sum(1 for u in us if u in fams["fabricated_geography"])
        h = sum(1 for u in us if u in fams["hour_attribution"])
        fl = sum(1 for u in us if u in fams["field_confusion"])
        ct = sum(1 for u in us if per_rec[u]["CONTRADICTED"] > 0)
        cl = sum(1 for u in us if u in clean)
        L.append(f"{cat:<32}{len(us):>4}{g:>6}{h:>6}{fl:>7}{ct:>8}{cl:>7}")

    report = "\n".join(L)
    print(report)
    tag = f"_{a.tag}" if a.tag else ""
    print()
    for fname, payload in (
            (f"stage_c_aggregate{tag}.txt", report),
            (f"stage_c_review{tag}.json",
             json.dumps(review, ensure_ascii=False, indent=1)),
            (f"stage_c_invented_names{tag}.json",
             json.dumps(names.most_common(), ensure_ascii=False, indent=1))):
        (out_dir / fname).write_text(payload, encoding="utf-8")
        print(f"-> {out_dir / fname}")


if __name__ == "__main__":
    main()
