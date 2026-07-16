"""Phase 5 Stage 3: the WFEDS decision-advisor agent (channel-agnostic).

Our own tool-calling loop over LiteLLM (model = config, see llm_config.py): the
LLM reads the user's free text + the pins the CHANNEL collected (it never guesses
coordinates), extracts the parameters (window/horizon/date), calls the ONE
deterministic tool (`run_scenario`) and then narrates the result + recommends
response measures in plain Greek. Suppression is advisory, never simulated.

The channel (Telegram / terminal / Dify) supplies: user text, the collected pins,
and a progress callback. The agent returns (reply_text, files_to_send).

Terminal smoke test (needs a working LLM key in .env):
    python scripts/agent/agent.py "τι θα κάψει σε 6 ώρες;" --pins "38.90,23.12"
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).parents[1] / "cell2fire"))

from llm_config import get_model  # noqa: E402

# User-facing times are Greece local (the analyst's wall clock); the engine
# speaks naive-UTC strings. tzdata comes in transitively via pandas.
ATHENS_TZ = ZoneInfo("Europe/Athens")

# NB: the "~375 μ." in rule 1 below is a HARDCODED literal mirroring
# build_cell2fire_instance.VIIRS_PIXEL_M (the ONE seeding rule: every pin is an
# observation buffered by half a VIIRS pixel) - kept in sync manually on
# purpose (importing the constant here would pull geopandas/rasterio into
# agent startup; this module only lazy-imports that file inside _run_tool).
SYSTEM_PROMPT = """\
Είσαι ο WFEDS, σύμβουλος αποφάσεων εκκένωσης για δασικές πυρκαγιές (Β. Εύβοια).
Διαβάζεις μια ΝΤΕΤΕΡΜΙΝΙΣΤΙΚΗ πρόβλεψη (Cell2Fire, ελεύθερο κάψιμο = χειρότερη
εύλογη περίπτωση) και συμβουλεύεις τον αναλυτή. ΔΕΝ προσομοιώνεις κατάσβεση - την
προτείνεις ως μέτρο όπου είναι εφικτή.

ΚΑΝΟΝΕΣ:
1. ΠΟΤΕ μη μαντεύεις συντεταγμένες. Η γεωμετρία έρχεται ΜΟΝΟ από τα pins του
   χρήστη (σου δίνονται). Κάθε pin είναι ΠΑΡΑΤΗΡΗΣΗ φωτιάς με φυσικό αποτύπωμα
   ~375 μ. (buffer μισού VIIRS pixel)· pins έως ~375 μ. μεταξύ τους ενώνονται
   γεωμετρικά σε ΕΝΑ μέτωπο, πιο μακρινά γίνονται ΞΕΧΩΡΙΣΤΕΣ, ανεξάρτητες
   φωτιές (δες n_fronts_detected στο αποτέλεσμα) - ανέφερέ το φυσικά στον
   χρήστη όταν συμβεί. Αν δεν υπάρχουν pins και ο χρήστης δεν ζητά το
   σενάριο αναφοράς, ζήτα του να στείλει τοποθεσία (📎 -> Location στο Telegram).
2. Από το κείμενο βγάζεις: ορίζοντα (ώρες), παράθυρο (km) και -αν δοθεί-
   ημερομηνία/ώρα έναρξης. Οι ώρες του χρήστη είναι ΤΟΠΙΚΗ ώρα Ελλάδας - πέρασέ
   τες αυτούσιες στο start_time (το εργαλείο τις μετατρέπει σε UTC). Για
   "τώρα"/ζωντανή φωτιά ή όταν δεν αναφέρεται ώρα, ΜΗΝ ρωτάς τον χρήστη την
   ώρα: παράλειψε το start_time και το εργαλείο βάζει μόνο του την τρέχουσα.
   Αν λείπουν παράθυρο/ορίζοντας, πρότεινε λογικές τιμές (παράθυρο 12 km,
   ορίζοντας 6 ώρες) και ΡΩΤΑ πριν τρέξεις - εκτός αν ο χρήστης είπε "τρέξε"
   γενικά, οπότε χρησιμοποίησέ τες. Για τωρινή/μελλοντική έναρξη χρησιμοποιείται
   πρόγνωση καιρού (έως ~15 μέρες μπροστά). ΠΕΣ ρητά όταν ο καιρός προέρχεται
   (έστω εν μέρει) από πρόγνωση (weather_source στο αποτέλεσμα) - είναι λιγότερο
   σίγουρος από το ιστορικό αρχείο.
3. "Επικύρωση" / "ιστορική φωτιά Β. Εύβοιας" -> scenario="north_evia_2021"
   (χωρίς pins/παράθυρο - είναι κλειδωμένα). Προσοχή: αργεί ~15 λεπτά.
4. Μετά το τρέξιμο: πες σε απλή γλώσσα τι θα συμβεί ανά ώρα (πότε κόβονται
   δρόμοι, ποιοι οικισμοί κινδυνεύουν/αποκλείονται), πρότεινε ΜΕΤΡΑ: σειρά και
   χρονισμό εκκένωσης, πού αξίζει αναχαίτιση (χαμηλό final_front_class4_pct =
   πιο μαχητό μέτωπο· ψηλό = μόνο έμμεση καταπολέμηση). Τα νούμερα είναι το
   χειρότερο σενάριο - πες το. Τα μέτρα που προτείνεις είναι ΓΝΩΜΟΔΟΤΙΚΑ,
   βασισμένα σε μοντέλο ελεύθερης καύσης (χωρίς κατάσβεση) και σε αβαθμονόμητα
   καύσιμα: ΠΑΝΤΑ κλείνε με μια σύντομη δήλωση ορίων - τι δεν ξέρει το μοντέλο
   (κατάσβεση, κίνηση οχημάτων, τοπικές συνθήκες) και ότι η τελική απόφαση
   ανήκει στον επιχειρησιακό υπεύθυνο, όχι στο μοντέλο. ΠΟΤΕ μη διατυπώνεις
   τα μέτρα ως απόλυτη σύσταση.
5. Αν το εργαλείο απορρίψει είσοδο (INPUT ERROR), μετέφερε το μήνυμα αυτούσιο
   και βοήθησε τον χρήστη να το διορθώσει.
6. Απαντάς ΠΑΝΤΑ στα ελληνικά, σύντομα και επιχειρησιακά. Μην επινοείς αριθμούς
   που δεν υπάρχουν στο αποτέλεσμα του εργαλείου.
7. Μετά από κάθε τρέξιμο, ο χρήστης βλέπει ΗΔΗ δύο δομημένες κάρτες (παράμετροι
   + αριθμητικά αποτελέσματα). ΜΗΝ επαναλαμβάνεις τα ίδια νούμερα σε λίστα -
   εστίασε στην ΕΡΜΗΝΕΙΑ: το χρονικό της εξέλιξης (πότε αλλάζει κάτι), τι
   σημαίνει για αποφάσεις, και τα προτεινόμενα μέτρα, σε 2-4 σύντομα bullets.
"""

TOOLS = [{
    "type": "function",
    "function": {
        "name": "run_fire_scenario",
        "description": ("Τρέχει την πλήρη ντετερμινιστική αλυσίδα: προσομοίωση "
                        "φωτιάς (ελεύθερο κάψιμο) + εκκένωση ανά ώρα + dashboard. "
                        "Είτε με τα pins του χρήστη (points), είτε με named "
                        "scenario. Επιστρέφει σύνοψη ανά ώρα + αρχεία."),
        "parameters": {
            "type": "object",
            "properties": {
                "use_pins": {"type": "boolean",
                             "description": "true = χρησιμοποίησε τα pins του χρήστη"},
                "window_km": {"type": "number", "description": "παράθυρο 6-40 km"},
                "horizon_h": {"type": "integer", "description": "ορίζοντας 1-48 ώρες"},
                "start_time": {"type": "string",
                               "description": ("YYYY-MM-DDTHH:MM τοπική ώρα "
                                               "Ελλάδας (μετατρέπεται σε UTC "
                                               "αυτόματα)· παράλειψέ το για "
                                               "«τώρα»")},
                "scenario": {"type": "string",
                             "description": "north_evia_2021 = το σενάριο επικύρωσης"},
                "label": {"type": "string",
                          "description": "σύντομη ετικέτα για το dashboard"},
            },
        },
    },
}]


def _to_utc(start_local):
    """User times (Greece local wall clock) -> the engine's naive-UTC string.
    None/"" -> the current UTC minute (the "live fire" default). A time that
    already carries an offset (e.g. ...+00:00 / Z) is respected as given.
    Raises ValueError on unparseable input (relayed as INPUT ERROR)."""
    if not start_local:
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M")
    t = datetime.fromisoformat(str(start_local).strip().replace("/", "-"))
    if t.tzinfo is None:
        t = t.replace(tzinfo=ATHENS_TZ)
    return t.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M")


def _params_card(result):
    """Deterministic settings card - the same structure on every run."""
    inp = result["inputs"]
    grid = inp.get("grid", {})
    mode = {"point_ignition": "1 pin (αποτύπωμα ~375 μ.)",
            "front": f"μέτωπο ({inp.get('front_cells', '?')} κελιά)",
            "multi_front": f"{inp.get('n_fronts_detected', '?')} ανεξάρτητα μέτωπα "
                           f"({inp.get('front_cells', '?')} κελιά συνολικά)",
            "scenario": f"σενάριο αναφοράς: {inp.get('scenario')}"}.get(
                inp.get("mode"), inp.get("mode", "?"))
    pts = inp.get("ignition_points_wgs84")
    pts_line = ("\n• Σημεία: " + " · ".join(f"{p[0]:.4f},{p[1]:.4f}" for p in pts)
                if isinstance(pts, list) else "")
    wkm = inp.get("window_km")
    win = (f"{wkm:g} km" if wkm else "VIIRS bbox") + \
          f" ({grid.get('nrows', '?')}×{grid.get('ncols', '?')} κελιά των 30 μ.)"
    t_utc = str(inp.get("start_time_utc", ""))
    try:                       # show the user's clock, keep UTC for the record
        t_loc = (datetime.fromisoformat(t_utc).replace(tzinfo=timezone.utc)
                 .astimezone(ATHENS_TZ))
        start_disp = (f"{t_loc:%Y-%m-%d %H:%M} ώρα Ελλάδας "
                      f"({t_utc.replace('T', ' ')} UTC)")
    except ValueError:
        start_disp = t_utc.replace("T", " ")
    wsrc = {"archive": "ιστορικό ERA5/Open-Meteo",
            "forecast": "ΠΡΟΓΝΩΣΗ Open-Meteo (λιγότερο σίγουρη από το ιστορικό αρχείο)",
            "archive+forecast": "ιστορικό + ΠΡΟΓΝΩΣΗ Open-Meteo (μεταβατικό, λιγότερο "
                                "σίγουρο για το προγνωστικό κομμάτι)"}.get(
                inp.get("weather_source"), "Open-Meteo")
    return ("⚙️ ΠΑΡΑΜΕΤΡΟΙ ΣΕΝΑΡΙΟΥ\n"
            f"• Τύπος: {mode}{pts_line}\n"
            f"• Παράθυρο: {win}\n"
            f"• Ορίζοντας: {inp.get('horizon_h')} ώρες\n"
            f"• Έναρξη: {start_disp}\n"
            f"• Καιρός: {wsrc}, άνεμος '{inp.get('wind_dir_source')}', "
            f"FWI υπολογισμένο (spin-up {inp.get('fwi_spinup_start')})\n"
            f"• Μοντέλο: ελεύθερο κάψιμο (χειρότερο σενάριο, χωρίς κατάσβεση)\n"
            f"• Run: {result['run_id']} · {result['elapsed_min']} λεπτά")


def _result_card(result):
    """Deterministic final-state card."""
    f = result.get("final_hour") or {}
    pct4 = result.get("final_front_class4_pct")
    lines = [f"📊 ΑΠΟΤΕΛΕΣΜΑ - τελική ώρα +{f.get('period', '?')}",
             f"🔥 Καμένη έκταση: {f.get('fire_km2', '?')} km²",
             f"🏘️ Σε κίνδυνο: {f.get('at_risk', 0)} οικισμοί "
             f"(~{f.get('population', 0):,} κάτοικοι)".replace(",", "."),
             f"🚗 Εκκενώνονται: {f.get('routed', 0)} · "
             f"⛔ Αποκλεισμένοι: {f.get('cut_off', 0)} · "
             f"🔥 Στο μέτωπο: {f.get('impacted', 0)}",
             f"🛣️ Κλειστά τμήματα δρόμων: {f.get('edges_removed', 0)}"]
    if f.get("longest_route_km"):
        lines.append(f"📏 Μεγαλύτερη διαδρομή: {f['longest_route_km']} km")
    if pct4 is not None:
        lines.append(f"⚔️ Μέτωπο κλάσης 4 (μόνο έμμεση καταπολέμηση): {pct4}%")
    return "\n".join(lines)


def _compact(result):
    """The tool result the LLM sees - result.json minus the bulky per-hour list
    (key deltas only), so small models don't drown."""
    per_hour = result.get("per_hour", [])
    events = []
    prev = None
    for row in per_hour:
        if prev is None or row["cut_off"] != prev["cut_off"] or \
                row["at_risk"] != prev["at_risk"]:
            events.append({k: row[k] for k in
                           ("period", "fire_km2", "at_risk", "population",
                            "routed", "cut_off", "impacted")})
        prev = row
    return {
        "run_id": result["run_id"],
        "inputs": {k: result["inputs"].get(k) for k in
                   ("mode", "window_km", "start_time_utc", "horizon_h",
                    "front_cells", "n_fronts_detected", "scenario",
                    "weather_source")},
        "final_hour": result["final_hour"],
        "final_front_class4_pct": result.get("final_front_class4_pct"),
        "change_hours": events,
        "elapsed_min": result["elapsed_min"],
    }


class WfedsAgent:
    """One conversation. The channel feeds text+pins; gets (reply, files)."""

    def __init__(self, preset=None):
        self.model, self.api_key = get_model(preset)
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    def chat(self, user_text, pins=None, progress_cb=None):
        import litellm

        pin_note = (f"[ΤΡΕΧΟΝΤΑ PINS ΧΡΗΣΤΗ: {pins}]" if pins
                    else "[ΚΑΝΕΝΑ PIN - ζήτα τοποθεσία αν χρειάζεται]")
        self.messages.append({"role": "user",
                              "content": f"{pin_note}\n{user_text}"})
        files, cards = [], []
        for _ in range(4):                      # tool-call rounds ceiling
            resp = litellm.completion(model=self.model, api_key=self.api_key,
                                      messages=self.messages, tools=TOOLS)
            msg = resp.choices[0].message
            self.messages.append(msg.model_dump(exclude_none=True))
            if not getattr(msg, "tool_calls", None):
                return (msg.content or "").strip(), files, cards
            for tc in msg.tool_calls:
                args = json.loads(tc.function.arguments or "{}")
                out, fs, cs = self._run_tool(args, pins, progress_cb)
                files.extend(fs)
                cards.extend(cs)
                self.messages.append({"role": "tool",
                                      "tool_call_id": tc.id,
                                      "content": out})
        return "Σταμάτησα - πολλές διαδοχικές κλήσεις εργαλείου.", files, cards

    def _run_tool(self, args, pins, progress_cb):
        from build_cell2fire_instance import (InstanceInputError,
                                              _validate_user_inputs)
        from run_scenario import run_scenario

        points = pins if (args.get("use_pins") and pins) else None
        scenario = args.get("scenario")
        try:
            horizon_h = int(args.get("horizon_h") or 6)
        except (TypeError, ValueError):
            return (f"INPUT ERROR: μη έγκυρος ορίζοντας: "
                    f"{args.get('horizon_h')!r}.", [], [])
        start_utc = None                  # scenario runs keep their locked start
        if not scenario:
            try:
                start_utc = _to_utc(args.get("start_time"))
            except ValueError:
                return (f"INPUT ERROR: μη αναγνωρίσιμη ημερομηνία/ώρα "
                        f"{args.get('start_time')!r} - χρειάζομαι "
                        f"YYYY-MM-DDTHH:MM (ώρα Ελλάδας).", [], [])
            # Fail fast: don't post "τρέχω..." for input the engine would
            # reject instantly (build_instance re-runs the same check later).
            try:
                _validate_user_inputs(points, args.get("window_km"),
                                      horizon_h, start_utc)
            except InstanceInputError as e:
                return f"INPUT ERROR: {e}", [], []
            except Exception as e:            # validation infrastructure failure
                return f"PIPELINE ERROR: {type(e).__name__}: {e}", [], []
        if progress_cb:
            eta = "~15 λεπτά" if scenario else "~1-3 λεπτά"
            progress_cb(f"⏳ Τρέχω το σενάριο ({eta})...")
        try:
            result = run_scenario(
                ignition_points=points,
                window_km=args.get("window_km"),
                horizon_h=horizon_h,
                start_time=start_utc,
                scenario=scenario,
                label=args.get("label"))
            files = [result["files"]["map_png"], result["files"].get("map_anim", ""),
                     result["files"]["dashboard"]]
            files = [f for f in files if f and Path(f).exists()]
            cards = [_params_card(result), _result_card(result)]
            return json.dumps(_compact(result), ensure_ascii=False), files, cards
        except InstanceInputError as e:
            return f"INPUT ERROR: {e}", [], []
        except Exception as e:                        # engine/pipeline failure
            return f"PIPELINE ERROR: {type(e).__name__}: {e}", [], []


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("text")
    ap.add_argument("--pins", default=None, help='"lat,lon[;lat,lon...]"')
    ap.add_argument("--preset", default=None)
    a = ap.parse_args()
    pins = ([tuple(float(x) for x in p.split(","))
             for p in a.pins.split(";")] if a.pins else None)
    agent = WfedsAgent(a.preset)
    reply, files, cards = agent.chat(a.text, pins, progress_cb=print)
    for c in cards:
        print("\n" + c)
    print("\n" + reply)
    for f in files:
        print(f"[file] {f}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
