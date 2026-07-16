"""Phase 5 Stage 4: the Telegram skin over the WFEDS agent.

Runs LOCALLY (long-polling - no server/public IP; the engine lives in this PC's
WSL). The user:

    📎 -> Location            drops pin(s): 1 = σημειακή έναυση; >=2 merge into
                              front(s) only when <= ~375 m apart (VIIRS pixel)
    plain text                talks to the agent ("τι θα κάψει σε 6 ώρες;")
    /clear                    forget pins + conversation
    /pins                     show the collected pins

The agent (agent.py) decides when to call the deterministic tool; the bot posts
progress, then the reply + the dashboard HTML as a document.

Needs in .env:  TELEGRAM_BOT_TOKEN=...   (BotFather) + a working LLM key.
Run:            python scripts/agent/telegram_bot.py
"""

import asyncio
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parents[1] / "cell2fire"))

from llm_config import _load_dotenv  # noqa: E402
from agent import WfedsAgent  # noqa: E402

from telegram import BotCommand, ReplyKeyboardMarkup, Update  # noqa: E402
from telegram.constants import ChatAction  # noqa: E402
from telegram.ext import (Application, CommandHandler, MessageHandler,  # noqa: E402
                          ContextTypes, filters)

WELCOME = (
    "🔥 *WFEDS* - σύμβουλος αποφάσεων εκκένωσης (Β. Εύβοια)\n\n"
    "1) Στείλε τοποθεσία (📎 -> Location): 1 pin = σημείο έναυσης, "
    "κοντινά pins (έως ~375 μ.) = μέτωπο\n"
    "2) Γράψε τι θες: π.χ. «τι θα κάψει σε 6 ώρες;»\n\n"
    "Θα πάρεις: κάρτες παραμέτρων/αποτελεσμάτων, ανάλυση + μέτρα, "
    "χάρτη, GIF εξέλιξης και το dashboard."
)

COMMANDS = [
    BotCommand("start", "έναρξη + οδηγίες"),
    BotCommand("help", "πώς δουλεύει, παραδείγματα"),
    BotCommand("pins", "τα σημεία που έχεις στείλει"),
    BotCommand("clear", "καθάρισμα pins + συζήτησης"),
    BotCommand("evia", "επικύρωση: ιστορική φωτιά Β. Εύβοιας 2021 (~15 λεπτά)"),
]

MENU_KB = ReplyKeyboardMarkup([["/pins", "/clear"], ["/help", "/evia"]],
                              resize_keyboard=True, is_persistent=True)

HELP = (
    "ℹ️ *Πώς δουλεύει*\n\n"
    "📍 *Γεωμετρία:* στείλε τοποθεσίες (📎 -> Location). 1 pin = σημειακή "
    "έναυση· pins έως ~375 μ. μεταξύ τους ενώνονται σε ΕΝΑ μέτωπο, πιο "
    "μακρινά γίνονται ξεχωριστές, ανεξάρτητες φωτιές.\n\n"
    "💬 *Παραδείγματα:*\n"
    "• «τι θα κάψει σε 6 ώρες;»\n"
    "• «τρέξε 12 ώρες με παράθυρο 20 km»\n"
    "• «ίδιο σημείο για τις 10 Αυγούστου 2024 το μεσημέρι»\n"
    "• «τρέξε την επικύρωση για την ιστορική φωτιά» (ή /evia)\n\n"
    "🕒 *Ώρες:* ό,τι γράφεις = ώρα Ελλάδας. Αν δεν δώσεις ώρα, "
    "χρησιμοποιείται η τωρινή («ζωντανή» φωτιά).\n"
    "⏱️ Ο καιρός: ιστορικό αρχείο (ERA5) για παλιότερες ημερομηνίες, "
    "πρόγνωση (έως ~15 μέρες μπροστά) για «τώρα»/ζωντανή φωτιά - δουλεύουν "
    "και τα δύο, η πρόγνωση είναι απλώς λιγότερο σίγουρη.\n"
    "🗺️ Παίρνεις: κάρτες, ανάλυση + μέτρα, χάρτη PNG, GIF εξέλιξης, dashboard "
    "HTML (για υπολογιστή).\n\n"
    "/pins - τα σημεία σου · /clear - καθάρισμα"
)

# NB: no distance check is possible right after a scenario=... run (which never
# touches pins) followed by a fresh pin - known, accepted limitation (there is
# nothing stored to compare against). Also in-memory only: a bot restart clears
# everything, so the stale-pin warning has no basis until the 2nd pin post-restart.
pins: dict[int, list] = {}
agents: dict[int, WfedsAgent] = {}


def _km_between(p1, p2):
    """Distance (km) between two (lat, lon) WGS84 points - flat-Earth approximation
    (same technique as meteo.py's IDW distance calc); adequate at this scale.
    Kept local/stdlib-only to keep this file dependency-light (no shapely/geopandas)."""
    lat1, lon1 = p1
    lat2, lon2 = p2
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = 111_320.0 * math.cos(math.radians((lat1 + lat2) / 2))
    dx = (lon2 - lon1) * m_per_deg_lon
    dy = (lat2 - lat1) * m_per_deg_lat
    return math.hypot(dx, dy) / 1000.0


def _n_fires(points, join_m):
    """How many independent fires the engine will see for these pins: centres
    <= join_m apart merge into one front (build_cell2fire_instance buffers each
    pin by half a VIIRS pixel and unions) - transitive, so a chain of close
    pins is ONE front. Connected components via union-find."""
    parent = list(range(len(points)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            if _km_between(points[i], points[j]) * 1000.0 <= join_m:
                parent[find(i)] = find(j)
    return len({find(i) for i in range(len(points))})


def _agent(chat_id):
    if chat_id not in agents:
        agents[chat_id] = WfedsAgent()
    return agents[chat_id]


async def cmd_start(update: Update, _ctx):
    await update.message.reply_text(WELCOME, parse_mode="Markdown",
                                    reply_markup=MENU_KB)


async def cmd_help(update: Update, _ctx):
    await update.message.reply_text(HELP, parse_mode="Markdown",
                                    reply_markup=MENU_KB)


async def cmd_clear(update: Update, _ctx):
    chat = update.effective_chat.id
    pins.pop(chat, None)
    agents.pop(chat, None)
    await update.message.reply_text("🧹 Καθάρισα pins και συζήτηση.")


async def cmd_pins(update: Update, _ctx):
    p = pins.get(update.effective_chat.id, [])
    if not p:
        await update.message.reply_text("Κανένα pin. Στείλε τοποθεσία (📎 -> Location).")
    else:
        rows = "\n".join(f"{i + 1}. {lat:.4f}, {lon:.4f}" for i, (lat, lon) in enumerate(p))
        await update.message.reply_text(
            f"📍 {len(p)} pin(s):\n{rows}\n\n"
            + ("(1 pin = σημειακή έναυση)" if len(p) == 1 else "(>=2 pins = μέτωπο)"))


async def on_location(update: Update, _ctx):
    chat = update.effective_chat.id
    loc = update.message.location
    new_pin = (round(loc.latitude, 5), round(loc.longitude, 5))
    # Stale-pin safety net: if the new pin is farther from an existing pin than
    # the pipeline's max simulation window, it very likely belongs to a DIFFERENT
    # fire (old pins forgotten without /clear). Warn, don't block (fail-open) -
    # the authoritative gate is build_instance's own window guardrail at run time.
    # No await between the read and the append below, so no interleaving hazard
    # (asyncio preempts only at await; PTB also processes updates sequentially).
    existing = pins.get(chat, [])
    far_km = None
    if existing:
        from build_cell2fire_instance import (WINDOW_KM_MAX,  # lazy, heavy deps
                                              VIIRS_PIXEL_M)
        d = max(_km_between(new_pin, p) for p in existing)
        if d > WINDOW_KM_MAX:
            far_km = d
    pins.setdefault(chat, []).append(new_pin)
    n = len(pins[chat])
    if n == 1:
        geom = "Ένα pin = σημειακή έναυση. "
    else:
        # Tell the user NOW how the engine will read the geometry - not after
        # the run (the "front" they think they drew may be N separate fires).
        k = _n_fires(pins[chat], VIIRS_PIXEL_M)
        geom = (f"{n} pins = ένα ενιαίο μέτωπο. " if k == 1 else
                f"{n} pins = {k} ξεχωριστές φωτιές - ενώνονται μόνο pins έως "
                f"~{VIIRS_PIXEL_M:.0f} μ. μεταξύ τους· για ενιαίο μέτωπο "
                f"στείλε ενδιάμεσα pins. ")
    await update.message.reply_text(
        f"📍 Pin {n} καταχωρήθηκε ({loc.latitude:.4f}, {loc.longitude:.4f}). "
        + geom + "Γράψε τι θες να δούμε (π.χ. «τι θα κάψει σε 6 ώρες;»).")
    if far_km is not None:
        # (WINDOW_KM_MAX is already bound: far_km is only set inside the branch
        # that imported it.)
        await update.message.reply_text(
            f"⚠️ Αυτό το σημείο απέχει ~{far_km:.0f} km από προηγούμενα pins σου - "
            f"μάλλον δεν χωράνε στο ίδιο παράθυρο προσομοίωσης (μέγιστο "
            f"{WINDOW_KM_MAX:.0f} km). Αν είναι άσχετη φωτιά, στείλε /clear πρώτα "
            f"και μετά νέο pin.")


async def on_text(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await _process_text(update, ctx, update.message.text)


async def cmd_evia(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await _process_text(update, ctx,
                        "Τρέξε την επικύρωση για την ιστορική φωτιά της Β. "
                        "Εύβοιας 2021 (scenario north_evia_2021).")


async def _process_text(update: Update, ctx: ContextTypes.DEFAULT_TYPE, text):
    chat = update.effective_chat.id
    agent = _agent(chat)
    loop = asyncio.get_running_loop()

    def progress(msg):
        asyncio.run_coroutine_threadsafe(
            ctx.bot.send_message(chat_id=chat, text=msg), loop)

    await ctx.bot.send_chat_action(chat_id=chat, action=ChatAction.TYPING)
    try:
        reply, files, cards = await asyncio.to_thread(
            agent.chat, text, pins.get(chat) or None, progress)
    except Exception as e:
        await update.message.reply_text(f"⚠️ Σφάλμα: {type(e).__name__}: {e}")
        return
    for card in cards:                       # structured settings/result cards first
        await update.message.reply_text(card)
    if reply:
        # The LLM writes GitHub-style **bold**; Telegram's legacy Markdown mode
        # wants *bold*. Convert, try to render; on any parse error (odd markup,
        # stray underscores) fall back to plain text rather than failing.
        try:
            await update.message.reply_text(reply.replace("**", "*"),
                                            parse_mode="Markdown")
        except Exception:
            await update.message.reply_text(reply)
    for f in files:
        try:
            if f.endswith(".png"):                    # final map with the info panel
                with open(f, "rb") as fh:
                    await ctx.bot.send_photo(
                        chat_id=chat, photo=fh,
                        caption="🗺️ Τελικός χάρτης + παράμετροι (χρώμα = πότε φτάνει η φωτιά)")
            elif f.endswith((".gif", ".mp4")):        # hour-by-hour animation, inline
                with open(f, "rb") as fh:
                    await ctx.bot.send_animation(
                        chat_id=chat, animation=fh,
                        caption="▶️ Η εξέλιξη ώρα-ώρα")
            else:                                     # dashboard HTML - for desktop
                with open(f, "rb") as fh:
                    await ctx.bot.send_document(
                        chat_id=chat, document=fh, filename=Path(f).name,
                        caption="Το διαδραστικό dashboard - για υπολογιστή (άνοιξέ το στον browser).")
        except Exception as e:
            await update.message.reply_text(f"(δεν στάλθηκε το {Path(f).name}: {e})")


async def _post_init(app):
    """Register the command MENU (the '/' button next to the input field)."""
    await app.bot.set_my_commands(COMMANDS)


def main():
    _load_dotenv()
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        raise SystemExit("Set TELEGRAM_BOT_TOKEN in .env (create a bot via @BotFather).")
    app = Application.builder().token(token).post_init(_post_init).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("clear", cmd_clear))
    app.add_handler(CommandHandler("pins", cmd_pins))
    app.add_handler(CommandHandler("evia", cmd_evia))
    app.add_handler(MessageHandler(filters.LOCATION, on_location))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))
    print("WFEDS Telegram bot: polling... (Ctrl+C για τερματισμό)")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
