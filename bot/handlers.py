from __future__ import annotations

import logging
from typing import List

from telegram import (
    Update,
    InlineKeyboardMarkup, InlineKeyboardButton,
    ReplyKeyboardMarkup, KeyboardButton, ForceReply,
)
from telegram.ext import (
    Application, CommandHandler, MessageHandler, CallbackQueryHandler,
    ContextTypes, filters,
)

from config import WATCHER_TFS, WATCHER_INTERVAL_SEC
from bot.watcher import schedule_watcher_jobs

from services.state import get_favorites, add_favorite, remove_favorite
from services.market_data import search_symbols
from services.analyze import analyze_symbol
from services.signal_text import build_signal_message

# === True Trading ===
from services.true_trading import get_tt

# === MM v2 commands ===
from bot.mm_v2_commands import register_mm_v2_handlers
from bot.mm_v2_events_commands import register_mm_v2_events_handlers  # ✅ NEW

log = logging.getLogger(__name__)


# ------------ Кнопка "Меню" ------------
def _menu_keyboard() -> ReplyKeyboardMarkup:
    rows: List[List[KeyboardButton]] = [
        [KeyboardButton("/list"), KeyboardButton("/find")],
        [KeyboardButton("/check")],
        [KeyboardButton("/watch_on"), KeyboardButton("/watch_off")],
        [KeyboardButton("/watch_status")],

        [KeyboardButton("/tt_on"), KeyboardButton("/tt_off")],
        [KeyboardButton("/tt_status")],

        # MM v2 (manual)
        [KeyboardButton("/mm_run")],
        [KeyboardButton("/mm_events_backfill")],  # ✅ NEW
    ]
    return ReplyKeyboardMarkup(rows, resize_keyboard=True)


# ------------ Вспомогательные клавиатуры ------------
def _favorites_inline_kb(symbols: List[str]) -> InlineKeyboardMarkup:
    rows = []
    for s in symbols:
        rows.append([
            InlineKeyboardButton(text=s, callback_data=f"sig:{s}"),
            InlineKeyboardButton(text="➖", callback_data=f"del:{s}"),
        ])
    if not rows:
        rows = [[InlineKeyboardButton(text="(список пуст)", callback_data="noop")]]
    return InlineKeyboardMarkup(rows)


def _search_results_kb(symbols: List[str]) -> InlineKeyboardMarkup:
    rows = []
    for s in symbols[:30]:
        rows.append([
            InlineKeyboardButton(text=s, callback_data=f"sig:{s}"),
            InlineKeyboardButton(text="➕", callback_data=f"add:{s}"),
        ])
    if not rows:
        rows = [[InlineKeyboardButton(text="ничего не найдено", callback_data="noop")]]
    return InlineKeyboardMarkup(rows)


# ------------ Команды ------------
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = (
        "Привет!\n"
        "• /list — избранные пары\n"
        "• /find ‹строка› — поиск пары\n"
        "• /check — анализ избранного\n"
        "• /watch_on — включить вотчер\n"
        "• /watch_off — выключить вотчер\n"
        "• /watch_status — статус вотчера\n"
        "• /tt_on — включить True Trading\n"
        "• /tt_off — выключить True Trading\n"
        "• /tt_status — статус True Trading\n"
        "• /mm_run — MM v2: ручной прогон (snapshots + regime + phase + live events)\n"
        "• /mm_events_backfill — MM v2: one-shot backfill событий за 30 дней\n"
        "• /menu — показать клавиатуру команд\n"
    )
    await update.message.reply_text(text, reply_markup=_menu_keyboard())


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Команды: /start, /help, /list, /find, /check, "
        "/watch_on, /watch_off, /watch_status, "
        "/tt_on, /tt_off, /tt_status, "
        "/mm_run, /mm_events_backfill, "
        "/menu"
    )


async def cmd_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("Меню команд:", reply_markup=_menu_keyboard())


async def cmd_list(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    favs = get_favorites()
    await update.message.reply_text("Избранные пары:", reply_markup=_favorites_inline_kb(favs))


async def cmd_find(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    q = " ".join(context.args).strip() if context.args else ""
    if q:
        syms = await search_symbols(q)
        await update.message.reply_text(
            f"Результаты по «{q}»:",
            reply_markup=_search_results_kb(syms),
        )
        return

    msg = await update.message.reply_text(
        "Напиши часть названия пары (например: btc или sol):",
        reply_markup=ForceReply(selective=True),
    )
    context.user_data["await_find_reply_to"] = msg.message_id


async def _on_text_find_reply(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    awaited_id = context.user_data.get("await_find_reply_to")
    if not awaited_id:
        return
    if not update.message or not update.message.reply_to_message:
        return
    if update.message.reply_to_message.message_id != awaited_id:
        return

    q = update.message.text.strip()
    context.user_data.pop("await_find_reply_to", None)
    if not q:
        await update.message.reply_text("Пустой запрос.")
        return

    syms = await search_symbols(q)
    await update.message.reply_text(
        f"Результаты по «{q}»:",
        reply_markup=_search_results_kb(syms),
    )


async def cmd_check(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    favs = get_favorites()
    if not favs:
        await update.message.reply_text("Список избранного пуст.")
        return

    await update.message.reply_text(f"Проверяю {len(favs)} пар…")
    for s in favs:
        try:
            res = await analyze_symbol(s)
            text = build_signal_message(res)
            await update.message.reply_text(text)
        except Exception as e:
            log.exception("check %s failed", s)
            await update.message.reply_text(f"{s}: ошибка анализа — {e}")


# ------------ Вотчер ------------
async def cmd_watch_on(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    created = schedule_watcher_jobs(
        app=context.application,
        tfs=WATCHER_TFS,
        interval_sec=int(WATCHER_INTERVAL_SEC),
    )
    tfs_txt = ", ".join([t for t in WATCHER_TFS]) or "—"
    await update.message.reply_text(
        f"Вотчер включён ✅\nTF: {tfs_txt}\ninterval={WATCHER_INTERVAL_SEC}s\njobs: {', '.join(created) or '—'}"
    )


async def cmd_watch_off(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    jq = context.application.job_queue
    removed = 0
    for job in list(jq.jobs()):
        if job and job.name and job.name.startswith("watch_"):
            try:
                job.schedule_removal()
                removed += 1
            except Exception:
                pass
    await update.message.reply_text(f"Вотчер выключен ⛔ (удалено jobs: {removed})")


async def cmd_watch_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    jq = context.application.job_queue
    jobs = [j for j in jq.jobs() if j and j.name and j.name.startswith("watch_")]

    if not jobs:
        await update.message.reply_text("Watcher: выключен ⛔")
        return

    lines = ["Watcher: включён ✅"]
    for j in sorted(jobs, key=lambda x: x.name):
        tf = j.name.replace("watch_", "", 1)
        nxt = getattr(j, "next_t", None)
        nxt_s = nxt.strftime("%Y-%m-%d %H:%M:%S UTC") if nxt else "—"
        lines.append(f"• TF {tf}: next={nxt_s}")
    await update.message.reply_text("\n".join(lines))


# ------------ True Trading команды ------------
async def cmd_tt_on(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    tt = get_tt(context.application)
    tt.enable()
    st = tt.status()
    txt = (
        "✅ True Trading включён.\n"
        f"Режим: REAL\n"
        f"Риск/сделку: {st.risk_pct*100:.2f}%\n"
        f"Лимит позиций: {st.max_open_pos}\n"
        f"Дневной лимит просадки: {st.daily_loss_limit_pct*100:.2f}%\n"
        f"Кулдаун по символу: {st.symbol_cooldown_min} мин.\n"
        f"Требуется 1D тренд: {'да' if st.require_trend_1d else 'нет'}\n"
        f"Мин RR до TP1: {st.min_rr_tp1:.2f}\n"
        f"Биржа подключена: {'да' if st.exchange_connected else 'нет (проверь ключи)'}"
    )
    await update.message.reply_text(txt)


async def cmd_tt_off(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    tt = get_tt(context.application)
    tt.disable()
    await update.message.reply_text("⛔ True Trading выключен. Ордеры не будут выставляться.")


async def cmd_tt_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    tt = get_tt(context.application)
    st = tt.status()
    since = st.since_ts
    txt = (
        "📟 Статус True Trading\n"
        f"Состояние: {'ВКЛ' if st.enabled else 'ВЫКЛ'}\n"
        f"С момента: {since if since else '—'} (unix)\n"
        f"Риск/сделку: {st.risk_pct*100:.2f}% | Мин RR: {st.min_rr_tp1:.2f}\n"
        f"Лимит позиций: {st.max_open_pos} | Дневной лимит: {st.daily_loss_limit_pct*100:.2f}%\n"
        f"Кулдаун по символу: {st.symbol_cooldown_min} мин | Slippage guard: {st.slippage_bps} б.п.\n"
        f"1D фильтр: {'да' if st.require_trend_1d else 'нет'} | Биржа: {'OK' if st.exchange_connected else 'нет ключей'}"
    )
    await update.message.reply_text(txt)


# ------------ Callback-кнопки ------------
async def on_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    q = update.callback_query
    if not q:
        return
    await q.answer()

    data = q.data or ""
    if data == "noop":
        return

    try:
        action, sym = data.split(":", 1)
        sym = sym.strip().upper()
    except Exception:
        return

    if action == "sig":
        try:
            res = await analyze_symbol(sym)
            text = build_signal_message(res)
            await q.message.reply_text(text)
        except Exception as e:
            log.exception("signal %s failed", sym)
            await q.message.reply_text(f"{sym}: ошибка анализа — {e}")

    elif action == "del":
        favs = remove_favorite(sym)
        await q.message.edit_text("Избранные пары:", reply_markup=_favorites_inline_kb(favs))

    elif action == "add":
        add_favorite(sym)
        await q.message.reply_text(f"{sym} добавлена в избранное ✅")


# ------------ Регистрация ------------
def register_handlers(app: Application) -> None:
    log.info(
        "Handlers зарегистрированы: /start, /help, /list, /find, /check, "
        "/watch_on, /watch_off, /watch_status, "
        "/tt_on, /tt_off, /tt_status, "
        "/mm_run, /mm_events_backfill, "
        "/menu"
    )

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("menu", cmd_menu))

    app.add_handler(CommandHandler("list", cmd_list))
    app.add_handler(CommandHandler("find", cmd_find))
    app.add_handler(CommandHandler("check", cmd_check))

    app.add_handler(CommandHandler("watch_on", cmd_watch_on))
    app.add_handler(CommandHandler("watch_off", cmd_watch_off))
    app.add_handler(CommandHandler("watch_status", cmd_watch_status))

    app.add_handler(CommandHandler("tt_on", cmd_tt_on))
    app.add_handler(CommandHandler("tt_off", cmd_tt_off))
    app.add_handler(CommandHandler("tt_status", cmd_tt_status))

    # MM v2 (manual + one-shot backfill)
    register_mm_v2_handlers(app)
    register_mm_v2_events_handlers(app)  # ✅ NEW

    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, _on_text_find_reply))
    app.add_handler(CallbackQueryHandler(on_callback))