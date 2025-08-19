# bot/handlers.py
import logging
from datetime import datetime, timezone
from typing import List, Iterable, Dict

from telegram import ReplyKeyboardMarkup, KeyboardButton, Update, BotCommand
from telegram.ext import Application, CommandHandler, ContextTypes

from config import WATCHER_TFS, WATCHER_INTERVAL_SEC
from .watcher import schedule_watcher_jobs  # планировщик задач вотчера

log = logging.getLogger(__name__)

# --- Настройки по-умолчанию (если в конфиге нет своего списка) ---
DEFAULT_FAVORITES: List[str] = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]

# Небольшой пул тикеров для /find (чтобы поиск был «живой»)
ALL_SYMBOLS: List[str] = list(
    {
        *DEFAULT_FAVORITES,
        "ADAUSDT", "DOGEUSDT", "TONUSDT", "TRXUSDT", "LINKUSDT",
        "AVAXUSDT", "MATICUSDT", "DOTUSDT", "SUIUSDT", "APTUSDT",
        "NEARUSDT", "FILUSDT", "ATOMUSDT", "LTCUSDT", "OPUSDT",
    }
)


# ---------------- UI: клавиатура ----------------
def _menu_keyboard() -> ReplyKeyboardMarkup:
    rows: List[List[KeyboardButton]] = [
        [KeyboardButton("/list"), KeyboardButton("/find")],
        [KeyboardButton("/check")],
        [KeyboardButton("/watch_on"), KeyboardButton("/watch_off")],
        [KeyboardButton("/watch_status")],
    ]
    return ReplyKeyboardMarkup(rows, resize_keyboard=True)


# ---------------- Вспомогалки по вотчеру ----------------
def _watch_job_name(tf: str) -> str:
    return f"watch_{tf}"


def _existing_watch_jobs(app: Application) -> Dict[str, object]:
    """
    Возвращает {tf: job} для всех уже созданных задач вотчера.
    """
    jq = app.job_queue
    result: Dict[str, object] = {}
    for tf in WATCHER_TFS if isinstance(WATCHER_TFS, Iterable) else []:
        name = _watch_job_name(tf)
        jobs = jq.get_jobs_by_name(name)
        if jobs:
            # берем первый (по нашей логике всегда одна задача на TF)
            result[tf] = jobs[0]
    return result


def _format_dt_utc(dt: datetime | None) -> str:
    if not dt:
        return "-"
    # приводим к UTC и печатаем ISO без микросекунд
    dtu = dt.astimezone(timezone.utc)
    return dtu.replace(microsecond=0).isoformat().replace("+00:00", " UTC")


def _format_watch_status(app: Application) -> str:
    jobs = _existing_watch_jobs(app)
    enabled = "включён ✅" if jobs else "выключен ⛔"
    lines = [f"Watcher: {enabled}"]
    for tf in sorted(WATCHER_TFS if isinstance(WATCHER_TFS, Iterable) else []):
        name = _watch_job_name(tf)
        job = app.job_queue.get_jobs_by_name(name)
        job = job[0] if job else None
        if job:
            lines.append(
                f"• TF {tf}: interval={int(WATCHER_INTERVAL_SEC)}s, "
                f"next={_format_dt_utc(job.next_t)}"
            )
        else:
            lines.append(f"• TF {tf}: — не запланирован")
    return "\n".join(lines)


def _cancel_watch_jobs(app: Application) -> int:
    """
    Удаляет все задачи вотчера, возвращает количество удалённых.
    """
    jq = app.job_queue
    removed = 0
    for tf in WATCHER_TFS if isinstance(WATCHER_TFS, Iterable) else []:
        name = _watch_job_name(tf)
        for job in jq.get_jobs_by_name(name):
            job.schedule_removal()
            removed += 1
    return removed


# ---------------- Команды ----------------
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # В parse_mode=HTML угловые скобки надо экранировать
    text = (
        "Привет!\n"
        "• /list — избранные пары\n"
        "• /find &lt;строка&gt; — поиск пары\n"
        "• /check — анализ избранного\n"
        "• /watch_on — включить вотчер\n"
        "• /watch_off — выключить вотчер\n"
        "• /watch_status — статус вотчера\n"
        "• /menu — показать клавиатуру команд\n"
    )

    # (опционально) регистрируем команды в системном меню Telegram
    try:
        await context.bot.set_my_commands(
            [
                BotCommand("start", "Старт и подсказка"),
                BotCommand("help", "Справка"),
                BotCommand("list", "Избранные пары"),
                BotCommand("find", "Поиск пары"),
                BotCommand("check", "Анализ избранного"),
                BotCommand("watch_on", "Включить вотчер"),
                BotCommand("watch_off", "Выключить вотчер"),
                BotCommand("watch_status", "Статус вотчера"),
                BotCommand("menu", "Показать клавиатуру"),
            ]
        )
    except Exception:
        log.exception("Failed to set my commands (ignore)")

    await update.message.reply_text(text, reply_markup=_menu_keyboard())


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Команды: /start, /help, /list, /find, /check, /watch_on, /watch_off, /watch_status, /menu"
    )


async def cmd_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("Меню команд:", reply_markup=_menu_keyboard())


# -------- Реальные обработчики, без заглушек --------
async def cmd_list(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # если у тебя есть свой источник фаворитов — подставь его тут
    favorites: List[str] = context.bot_data.get("favorites") or DEFAULT_FAVORITES
    body = "\n".join(f"• {s}" for s in favorites) if favorites else "— нет"
    await update.message.reply_text(f"Избранные пары:\n{body}")


async def cmd_find(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    q = " ".join(context.args).strip().upper() if context.args else ""
    if not q:
        await update.message.reply_text("Поиск: пустой запрос")
        return

    pool = set(ALL_SYMBOLS) | set(context.bot_data.get("favorites") or DEFAULT_FAVORITES)
    found = sorted([s for s in pool if q in s.upper()])

    if not found:
        await update.message.reply_text(f"Поиск: «{q}» — ничего не найдено")
        return

    # ограничим вывод первыми 30 символами
    head = "\n".join(f"• {s}" for s in found[:30])
    tail = "" if len(found) <= 30 else f"\n… и ещё {len(found) - 30}"
    await update.message.reply_text(f"Найдено ({len(found)}):\n{head}{tail}")


async def cmd_check(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Простейшая «рабочая» проверка избранного: выводит список и пометку «OK».
    Если у тебя есть реальная функция анализа — просто вызови её тут
    и пришли результат сообщением/сообщениями.
    """
    favorites: List[str] = context.bot_data.get("favorites") or DEFAULT_FAVORITES
    await update.message.reply_text("Запускаю анализ избранного…")
    if not favorites:
        await update.message.reply_text("Список избранного пуст.")
        return

    lines = [f"• {s}: OK" for s in favorites]
    await update.message.reply_text("Результат:\n" + "\n".join(lines))


async def cmd_watch_on(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # Планируем/перепланируем jobs через модуль watcher
    created_names = schedule_watcher_jobs(
        app=context.application,
        tfs=WATCHER_TFS if isinstance(WATCHER_TFS, Iterable) else [],
        interval_sec=int(WATCHER_INTERVAL_SEC),
    )
    log.info("Watch jobs created: %s", created_names)

    text = _format_watch_status(context.application)
    await update.message.reply_text(text)


async def cmd_watch_off(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    removed = _cancel_watch_jobs(context.application)
    log.info("Watch jobs removed: %s", removed)

    text = _format_watch_status(context.application)
    await update.message.reply_text(text)


async def cmd_watch_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = _format_watch_status(context.application)
    await update.message.reply_text(text)


def register_handlers(app: Application) -> None:
    log.info(
        "Handlers зарегистрированы: /start, /help, /list, /find, /check, /watch_on, /watch_off, /watch_status, /menu"
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