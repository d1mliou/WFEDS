"""Build the Stage-C audit workbook: every claim label in one .xlsx for the user
to check, with the tool's own numbers alongside so nothing has to be looked up.

The labels published in `stage_c_labels.json` are Claude's (protocol revised
2026-08-27, see [[Decision log]]). This workbook is the instrument for the human
audit that turns them from "LLM-labelled" into "LLM-labelled, human-audited".

Sheets are ordered by how much each label matters to the reported findings:
  1. Κατηγορίες   NOT_IN_RESULT + CONTRADICTED - every negative finding rests on
                  these, so they are checked one by one;
  2. Γνωμοδοτικά  ADVISORY_INFERENCE - the opposite risk (a fabrication excused
                  as grounded advice), sampled rather than exhausted;
  3. Υπόλοιπα     SUPPORTED + LIMITATION_STATEMENT - near-mechanical, sampled;
  4. Απαντήσεις   one row per repetition: question, reply, tool numbers, notes;
  5. Σύνοψη       the aggregate the thesis quotes, plus live audit progress;
  6. Ονόματα      the invented place/road name inventory.

Run:
    python scripts/validation/stage_c_audit_workbook.py <labels.json> <stage_c_full_dir> [--out FILE]
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation

LABELS = ["SUPPORTED", "ADVISORY_INFERENCE", "LIMITATION_STATEMENT",
          "NOT_IN_RESULT", "CONTRADICTED"]
ACCUSATION = {"NOT_IN_RESULT", "CONTRADICTED"}

HEAD_FILL = PatternFill("solid", fgColor="1F3864")
HEAD_FONT = Font(color="FFFFFF", bold=True, size=11)
ASK_FILL = PatternFill("solid", fgColor="FFF2CC")      # columns the user fills
TIER_FILL = {1: PatternFill("solid", fgColor="F8CBAD"),
             2: PatternFill("solid", fgColor="FFE699"),
             3: PatternFill("solid", fgColor="E2EFDA")}
THIN = Side(style="thin", color="BFBFBF")
BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)
TOP = Alignment(vertical="top", wrap_text=True)


def style_header(ws, widths: list[int], freeze: str) -> None:
    for i, w in enumerate(widths, 1):
        ws.column_dimensions[get_column_letter(i)].width = w
    for c in ws[1]:
        c.fill, c.font = HEAD_FILL, HEAD_FONT
        c.alignment = Alignment(vertical="center", wrap_text=True)
    ws.row_dimensions[1].height = 30
    ws.freeze_panes = freeze
    ws.auto_filter.ref = ws.dimensions


def claim_sheet(wb, title, rows, note):
    """rows: list of (uid, case_id, category, claim, label, justification).

    Row 1 is the tier note, row 2 the header, data from row 3 - built in that
    order because openpyxl does not shift data-validation ranges on insert_rows.
    """
    ws = wb.create_sheet(title)
    ws.append([note])
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=10)
    ws["A1"].font = Font(bold=True, size=11, color="833C00")
    ws["A1"].alignment = Alignment(vertical="center", wrap_text=True)
    ws.row_dimensions[1].height = 46

    ws.append(["#", "uid", "case", "κατηγορία", "ισχυρισμός (από την απάντηση)",
               "ετικέτα (Claude)", "αιτιολόγηση (Claude)",
               "ΣΩΣΤΗ;", "ΣΩΣΤΗ ΕΤΙΚΕΤΑ", "ΣΧΟΛΙΟ"])
    for i, r in enumerate(rows, 1):
        ws.append([i, *r, None, None, None])

    for i, w in enumerate([5, 9, 7, 20, 62, 20, 62, 10, 22, 34], 1):
        ws.column_dimensions[get_column_letter(i)].width = w
    for c in ws[2]:
        c.fill, c.font = HEAD_FILL, HEAD_FONT
        c.alignment = Alignment(vertical="center", wrap_text=True)
    ws.row_dimensions[2].height = 30

    n = ws.max_row
    for row in ws.iter_rows(min_row=3, max_row=n, max_col=10):
        for c in row:
            c.alignment, c.border = TOP, BORDER
        for c in row[7:]:
            c.fill = ASK_FILL
    if n > 2:
        yn = DataValidation(type="list", formula1='"ΝΑΙ,ΟΧΙ,ΑΒΕΒΑΙΟ"',
                            allow_blank=True)
        lab = DataValidation(type="list", formula1='"%s"' % ",".join(LABELS),
                             allow_blank=True)
        ws.add_data_validation(yn)
        ws.add_data_validation(lab)
        yn.add(f"H3:H{n}")
        lab.add(f"I3:I{n}")
    ws.freeze_panes = "E3"
    ws.auto_filter.ref = f"A2:J{n}"
    return ws


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("labels")
    ap.add_argument("batch_dir")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    batch_dir = Path(a.batch_dir)
    out = Path(a.out) if a.out else batch_dir / "stage_c_audit.xlsx"

    meta = {}
    for f in sorted(batch_dir.glob("batch_*.json")):
        for r in json.loads(f.read_text(encoding="utf-8")):
            meta[r["uid"]] = r
    recs = [r for r in json.loads(Path(a.labels).read_text(encoding="utf-8"))
            if r.get("uid") in meta]

    tier1, tier2, tier3, counts = [], [], [], Counter()
    per_rec = {}
    for r in recs:
        m = meta[r["uid"]]
        c = Counter()
        for x in r["claims"]:
            if not isinstance(x, dict) or x.get("label") not in LABELS:
                continue
            c[x["label"]] += 1
            row = (r["uid"], m["case_id"], m["category"], x.get("text", ""),
                   x["label"], x.get("justification", ""))
            if x["label"] in ACCUSATION:
                tier1.append(row)
            elif x["label"] == "ADVISORY_INFERENCE":
                tier2.append(row)
            else:
                tier3.append(row)
        counts.update(c)
        per_rec[r["uid"]] = c

    # heaviest records first, so the audit starts where the findings are densest
    weight = {u: per_rec[u]["NOT_IN_RESULT"] + per_rec[u]["CONTRADICTED"]
              for u in per_rec}
    tier1.sort(key=lambda r: (-weight[r[0]], r[0]))
    tier2.sort(key=lambda r: (-weight[r[0]], r[0]))
    tier3.sort(key=lambda r: r[0])

    wb = Workbook()
    wb.remove(wb.active)

    # ---------------------------------------------------------------- 0 guide
    ws = wb.create_sheet("Οδηγίες")
    ws.column_dimensions["A"].width = 118
    guide = [
        ("Έλεγχος ετικετών - Άξονας 3, Stage C", "title"),
        (f"{len(recs)} αφηγηματικές απαντήσεις, {sum(counts.values())} ισχυρισμοί. "
         "Οι ετικέτες είναι του Claude. Ο δικός σου έλεγχος τις κάνει "
         "«επισημειωμένες από LLM, ελεγμένες από άνθρωπο».", ""),
        ("", ""),
        ("Τι συμπληρώνεις (μόνο τα κίτρινα κελιά)", "h"),
        ("ΣΩΣΤΗ;  ->  ΝΑΙ / ΟΧΙ / ΑΒΕΒΑΙΟ. Αν ΟΧΙ, γράψε στη διπλανή στήλη ποια "
         "ετικέτα είναι η σωστή.", ""),
        ("Δεν χρειάζεται να γράψεις σχόλιο παντού. Μόνο όπου διαφωνείς ή "
         "δυσκολεύεσαι.", ""),
        ("", ""),
        ("Με ποια σειρά", "h"),
        ("1. «Κατηγορίες» - ΥΠΟΧΡΕΩΤΙΚΟ, μία μία. Κάθε αρνητικό εύρημα του άξονα "
         "στηρίζεται αποκλειστικά σε αυτές. Αν κάποια είναι λάθος, το εύρημα "
         "υπερβάλλει, και εκεί θα χτυπήσει ο εξεταστής.", ""),
        ("2. «Γνωμοδοτικά» - ΔΕΙΓΜΑ (~50 αρκεί). Ο αντίστροφος κίνδυνος: σύσταση "
         "που χαρακτηρίστηκε θεμιτή ενώ ήταν αστήρικτη. Αν υπάρχουν λάθη εδώ, τα "
         "ευρήματα ΥΠΟΤΙΜΟΥΝ το πρόβλημα.", ""),
        ("3. «Υπόλοιπα» - ΔΕΙΓΜΑ (~30 αρκεί). Σχεδόν μηχανικές.", ""),
        ("", ""),
        ("Πού βρίσκεις τα δεδομένα", "h"),
        ("Το φύλλο «Απαντήσεις» έχει, ανά uid: την ερώτηση του αναλυτή, ολόκληρη "
         "την απάντηση, ΤΟΥΣ ΑΡΙΘΜΟΥΣ ΠΟΥ ΟΝΤΩΣ ΕΠΕΣΤΡΕΨΕ ΤΟ ΕΡΓΑΛΕΙΟ, και τις "
         "σημειώσεις του επισημειωτή. Φιλτράρισε με το uid.", ""),
        ("", ""),
        ("Οι πέντε ετικέτες", "h"),
        ("SUPPORTED - ο ισχυρισμός αντιστοιχεί σε συγκεκριμένη τιμή του εργαλείου.", ""),
        ("CONTRADICTED - συγκρούεται με το εργαλείο. Μετράει και η σωστή τιμή "
         "κολλημένη σε λάθος ώρα ή σε λάθος πεδίο.", ""),
        ("NOT_IN_RESULT - δεν υπάρχει καν αντίστοιχο πεδίο. Εδώ πέφτουν τα "
         "επινοημένα ονόματα οικισμών και δρόμων, οι κατευθύνσεις, οι άνεμοι.", ""),
        ("ADVISORY_INFERENCE - σύσταση ή κρίση που (α) πατάει σε συγκεκριμένη τιμή "
         "και (β) συνοδεύεται από δήλωση ορίων. Αν λείπει κάποιο από τα δύο, είναι "
         "NOT_IN_RESULT.", ""),
        ("LIMITATION_STATEMENT - κομμάτι της υποχρεωτικής δήλωσης ορίων.", ""),
        ("", ""),
        ("ΚΡΙΣΙΜΟ: το εργαλείο ΔΕΝ επιστρέφει ποτέ όνομα οικισμού, όνομα δρόμου, "
         "κατεύθυνση, άνεμο, έδαφος, καύσιμα, πληθυσμό ανά οικισμό, ούτε ανάλυση "
         "ανά μέτωπο. Μόνο αθροιστικά πλήθη. Οτιδήποτε τέτοιο στην απάντηση είναι "
         "επινοημένο.", "warn"),
    ]
    for text, kind in guide:
        ws.append([text])
        c = ws.cell(row=ws.max_row, column=1)
        c.alignment = Alignment(vertical="top", wrap_text=True)
        if kind == "title":
            c.font = Font(bold=True, size=16, color="1F3864")
        elif kind == "h":
            c.font = Font(bold=True, size=12, color="1F3864")
        elif kind == "warn":
            c.font = Font(bold=True, color="9C0006")
            c.fill = PatternFill("solid", fgColor="FFC7CE")

    # -------------------------------------------------------------- 1-3 claims
    claim_sheet(
        wb, "1. Κατηγορίες", tier1,
        f"ΥΠΟΧΡΕΩΤΙΚΟ - {len(tier1)} ετικέτες σε {len({r[0] for r in tier1})} "
        "απαντήσεις. Ταξινομημένες ώστε πρώτα να έρχονται οι απαντήσεις με τις "
        "περισσότερες κατηγορίες. Κάθε αρνητικό νούμερο της διπλωματικής βγαίνει "
        "από εδώ.")
    claim_sheet(
        wb, "2. Γνωμοδοτικά", tier2,
        f"ΔΕΙΓΜΑ - {len(tier2)} ετικέτες. Δεν χρειάζεται να τις δεις όλες: ~50 "
        "στην τύχη αρκούν για να εκτιμήσεις πόσο ΥΠΟΤΙΜΟΥΝ τα ευρήματα.")
    claim_sheet(
        wb, "3. Υπόλοιπα", tier3,
        f"ΔΕΙΓΜΑ - {len(tier3)} ετικέτες (SUPPORTED και LIMITATION_STATEMENT). "
        "Σχεδόν μηχανικές, ~30 στην τύχη αρκούν.")

    # ---------------------------------------------------------- 4 repetitions
    ws = wb.create_sheet("Απαντήσεις")
    ws.append(["uid", "case", "κατηγορία", "ερώτηση αναλυτή",
               "απάντηση του agent (πλήρης)",
               "ΤΙ ΕΠΕΣΤΡΕΨΕ ΤΟ ΕΡΓΑΛΕΙΟ", "σημειώσεις επισημειωτή",
               *LABELS, "κατηγορίες"])
    for r in sorted(recs, key=lambda r: (-weight[r["uid"]], r["uid"])):
        m, c = meta[r["uid"]], per_rec[r["uid"]]
        ws.append([r["uid"], m["case_id"], m["category"], m["user_text"],
                   m["reply"], m.get("tool_numbers_md", ""), r.get("notes", ""),
                   *[c[l] for l in LABELS], weight[r["uid"]]])
    style_header(ws, [9, 7, 22, 46, 78, 62, 78, 11, 11, 11, 11, 11, 11], "D2")
    for row in ws.iter_rows(min_row=2, max_row=ws.max_row):
        for cell in row:
            cell.alignment, cell.border = TOP, BORDER

    # ------------------------------------------------------------- 5 summary
    ws = wb.create_sheet("Σύνοψη")
    ws.column_dimensions["A"].width = 52
    for w in ("B", "C", "D"):
        ws.column_dimensions[w].width = 15
    ws.append(["Stage C - σύνοψη", "", "", ""])
    ws["A1"].font = Font(bold=True, size=14, color="1F3864")
    ws.append([])
    ws.append(["μέγεθος", "τιμή", "", ""])
    ws.append(["αφηγηματικές επαναλήψεις", len(recs)])
    ws.append(["ατομικοί ισχυρισμοί", sum(counts.values())])
    ws.append([])
    ws.append(["ετικέτα", "πλήθος", "ποσοστό", ""])
    tot = sum(counts.values())
    for l in LABELS:
        ws.append([l, counts[l], round(100 * counts[l] / tot, 1)])
    ws.append([])
    ws.append(["ανά επανάληψη", "πλήθος", "ποσοστό", ""])
    n_nir = sum(1 for u in per_rec if per_rec[u]["NOT_IN_RESULT"] > 0)
    n_con = sum(1 for u in per_rec if per_rec[u]["CONTRADICTED"] > 0)
    n_cle = sum(1 for u in per_rec if weight[u] == 0)
    for name, k in (("με >=1 NOT_IN_RESULT", n_nir),
                    ("με >=1 CONTRADICTED", n_con),
                    ("χωρίς κανέναν αστήρικτο ισχυρισμό", n_cle)):
        ws.append([name, k, round(100 * k / len(recs), 1)])
    ws.append([])
    ws.append(["ΠΡΟΟΔΟΣ ΕΛΕΓΧΟΥ", "ελεγμένα", "διαφωνίες", "σύνολο"])
    for sheet, total in (("1. Κατηγορίες", len(tier1)),
                         ("2. Γνωμοδοτικά", len(tier2)),
                         ("3. Υπόλοιπα", len(tier3))):
        ws.append([sheet,
                   f"=COUNTA('{sheet}'!H3:H{total + 2})",
                   f"=COUNTIF('{sheet}'!H3:H{total + 2},\"ΟΧΙ\")",
                   total])
    for row in ws.iter_rows(min_row=1, max_row=ws.max_row, max_col=4):
        for cell in row:
            cell.alignment = Alignment(vertical="center", wrap_text=True)
        if isinstance(row[0].value, str) and row[0].value in (
                "μέγεθος", "ετικέτα", "ανά επανάληψη", "ΠΡΟΟΔΟΣ ΕΛΕΓΧΟΥ"):
            for cell in row:
                cell.fill, cell.font = HEAD_FILL, HEAD_FONT

    # --------------------------------------------------------------- 6 names
    inv = batch_dir / "stage_c_invented_names.json"
    if inv.exists():
        ws = wb.create_sheet("Επινοημένα ονόματα")
        ws.append(["επινοημένο όνομα", "σε πόσες επαναλήψεις"])
        for name, k in json.loads(inv.read_text(encoding="utf-8")):
            ws.append([name, k])
        style_header(ws, [48, 22], "A2")

    wb.save(out)
    print(f"{len(recs)} repetitions, {tot} labels")
    print(f"  1. Κατηγορίες  {len(tier1):5d}  (υποχρεωτικό)")
    print(f"  2. Γνωμοδοτικά {len(tier2):5d}  (δείγμα)")
    print(f"  3. Υπόλοιπα    {len(tier3):5d}  (δείγμα)")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
