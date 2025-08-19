# main.py — стабильный long polling без двойных запросов и без закрытия event loop
from __future__ import annotations
import asyncio
import logging
import os
from typing import Optional

from telegram.error import Conflict as TgConflict
from telegram.ext import Application, ApplicationBuilder

# ЛОГИ
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("main")

# ЧИТАЕМ ТОКЕН
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not TOKEN:
    raise RuntimeError("TELEGRAM_BOT_TOKEN не задан в переменных окружения")

# Путь к хендлерам (не обязателен — можно вызывать register_handlers, если он есть)
def try_register_handlers(app: Application) -> None:
    """
    Подключаем handlers, если модуль/символ есть.
    Не валим приложение, если файл в процессе разработки.
    """
    try:
        from bot.handlers import register_handlers  # type: ignore
    except Exception as e:
        log.warning("register_handlers не импортирован/не найден: %s", e)
        return
    try:
        register_handlers(app)
        log.info("Handlers зарегистрированы.")
    except Exception as e:
        log.error("Ошибка при регистрации handlers: %s", e)


async def build_app() -> Application:
    app = ApplicationBuilder().token(TOKEN).build()
    try_register_handlers(app)
    return app


async def start_polling(app: Application) -> None:
    """
    Одна итерация старта polling с корректным удалением вебхука.
    Ничего не закрывает, чтобы избежать 'Cannot close a running event loop'.
    """
    # Гарантированно удаляем вебхук, чтобы исключить конфликт polling/webhook
    try:
        await app.bot.delete_webhook(drop_pending_updates=True)
        log.info("Webhook удалён (drop_pending_updates=True).")
    except Exception as e:
        log.warning("Не удалось удалить webhook: %s", e)

    # Инициализация и старт приложения
    await app.initialize()
    await app.start()

    # Старт long polling (НЕ закрываем loop вручную!)
    await app.updater.start_polling(  # type: ignore[attr-defined]
        allowed_updates=None,  # все типы апдейтов
        timeout=30,            # long-poll таймаут
        read_timeout=35,       # запас по чтению
        write_timeout=35,
        connect_timeout=35,
        pool_timeout=35,
    )
    log.info("Polling запущен.")


async def idle(app: Application) -> None:
    """
    Ожидание до остановки (Ctrl+C/терминейт процесса платформой).
    """
    try:
        await app.updater.wait_until_closed()  # type: ignore[attr-defined]
    finally:
        # Аккуратно останавливаемся (без закрытия event loop вручную)
        try:
            await app.updater.stop()  # type: ignore[attr-defined]
        except Exception:
            pass
        try:
            await app.stop()
        except Exception:
            pass
        log.info("Остановка завершена.")


async def run_forever() -> None:
    """
    Главный цикл: держит один инстанс polling, переживает 409 Conflict с бэкоффом.
    """
    app: Optional[Application] = None

    while True:
        try:
            if app is None:
                app = await build_app()

            await start_polling(app)
            await idle(app)  # блокируется до остановки polling
            # Если вышли из idle без исключения — перезапустим цикл через паузу
            log.warning("Polling завершился без исключения. Перезапуск через 5 секунд.")
            await asyncio.sleep(5)

        except TgConflict as e:
            # Классический случай: второй getUpdates где-то ещё
            log.error("409 Conflict (кто-то ещё вызывает getUpdates на этом токене): %s", e)
            # Ничего не делаем с app; небольшой бэкофф и пробуем заново
            await asyncio.sleep(20)

        except Exception as e:
            log.exception("Неожиданная ошибка в polling цикле: %s", e)
            await asyncio.sleep(5)

        finally:
            # На всякий случай, если апдейтер запущен — мягко остановим
            if app is not None:
                try:
                    await app.updater.stop()  # type: ignore[attr-defined]
                except Exception:
                    pass
                try:
                    await app.stop()
                except Exception:
                    pass
                # НЕ закрываем loop руками!
                # Следующая итерация создаст Application заново, если нужно
                app = None


def main() -> None:
    log.info(">>> ENTER main.py")
    # Создаём (или берём) текущий event loop и гоняем нашу корутину
    loop = asyncio.get_event_loop()
    # Важно: НЕ закрывать loop в конце, чтобы не ловить "Cannot close a running event loop"
    loop.create_task(run_forever())
    try:
        loop.run_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()