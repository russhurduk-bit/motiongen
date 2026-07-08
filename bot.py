"""
MotionGen Telegram bot.
Flow:  /start → [🎬 Сделать видео] → upload photo → upload video
       → cost preview → [✅ Старт / ❌ Отмена] → generation → document+metrics.

Access: whitelist from ALLOWED_USER_IDS env (space-separated).
Only ONE generation runs at a time (global asyncio.Lock).
"""
import asyncio, os, sys, time, logging, traceback
from pathlib import Path

from dotenv import dotenv_values
from aiogram import Bot, Dispatcher, F, Router
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode, ChatAction
from aiogram.filters import Command, CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import (Message, CallbackQuery, FSInputFile, BufferedInputFile,
                            ReplyKeyboardMarkup, KeyboardButton,
                            InlineKeyboardMarkup, InlineKeyboardButton,
                            ReplyKeyboardRemove)

import engine
import sqlite3

BASE = Path(__file__).parent
ENV = dotenv_values(BASE / ".env")
BOT_TOKEN = ENV["BOT_TOKEN"]
ALLOWED = set(int(x) for x in (ENV.get("ALLOWED_USER_IDS") or "").split() if x)
DAILY_CREDIT_LIMIT = int(ENV.get("DAILY_CREDIT_LIMIT", "80"))   # per user per day

UPLOAD_DIR = BASE / "bot_uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
USAGE_DB = BASE / "usage.db"


def _usage_conn():
    c = sqlite3.connect(USAGE_DB)
    c.execute("""CREATE TABLE IF NOT EXISTS usage(
        user_id INTEGER, day TEXT, credits INTEGER DEFAULT 0,
        PRIMARY KEY(user_id, day))""")
    return c


def credits_used_today(user_id: int) -> int:
    import datetime as _dt
    day = _dt.date.today().isoformat()
    with _usage_conn() as c:
        row = c.execute("SELECT credits FROM usage WHERE user_id=? AND day=?",
                        (user_id, day)).fetchone()
    return row[0] if row else 0


def record_usage(user_id: int, credits: int):
    import datetime as _dt
    day = _dt.date.today().isoformat()
    with _usage_conn() as c:
        c.execute("INSERT INTO usage(user_id,day,credits) VALUES(?,?,?) "
                  "ON CONFLICT(user_id,day) DO UPDATE SET credits=credits+?",
                  (user_id, day, credits, credits))
        c.commit()

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("motiongen-bot")

bot = Bot(BOT_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.MARKDOWN))
dp = Dispatcher(storage=MemoryStorage())
router = Router()
dp.include_router(router)

GEN_LOCK = asyncio.Lock()  # одна генерация в один момент


class GenState(StatesGroup):
    awaiting_photo = State()
    awaiting_video = State()
    awaiting_confirm = State()


# ----------------------- helpers -------------------------------------

def is_allowed(user_id: int) -> bool:
    return user_id in ALLOWED


MAIN_KB = InlineKeyboardMarkup(inline_keyboard=[
    [InlineKeyboardButton(text="🎬 Higgsfield (Kling 3.0)", callback_data="new_gen:higgsfield")],
    [InlineKeyboardButton(text="⚡ Vast.ai (Wan 2.2, дёшево)", callback_data="new_gen:vast")],
    [InlineKeyboardButton(text="ℹ️ Как это работает", callback_data="help")],
])

CANCEL_KB = InlineKeyboardMarkup(inline_keyboard=[
    [InlineKeyboardButton(text="❌ Отмена", callback_data="cancel")],
])


def confirm_kb(cost: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text=f"✅ Запустить ({cost} кр)", callback_data="go")],
        [InlineKeyboardButton(text="❌ Отмена", callback_data="cancel")],
    ])


async def deny(msg: Message):
    await msg.answer(
        "🚫 Доступ закрыт.\n"
        f"Твой ID: `{msg.from_user.id}`, @{msg.from_user.username or '-'}\n"
        "Попроси администратора добавить тебя.")


# ----------------------- handlers ------------------------------------

@router.message(Command("stats"))
async def cmd_stats(msg: Message):
    if not is_allowed(msg.from_user.id):
        await deny(msg); return
    import datetime as _dt
    today = _dt.date.today().isoformat()
    with _usage_conn() as c:
        rows = c.execute(
            "SELECT user_id, credits FROM usage WHERE day=? ORDER BY credits DESC",
            (today,)).fetchall()
        totals = c.execute(
            "SELECT user_id, SUM(credits) FROM usage GROUP BY user_id "
            "ORDER BY 2 DESC").fetchall()
    txt = f"*Кредиты за сегодня ({today}):*\n"
    txt += ("\n".join(f"• `{uid}` — {cr} кр" for uid, cr in rows) or "пусто")
    txt += "\n\n*Всего за всё время:*\n"
    txt += ("\n".join(f"• `{uid}` — {cr or 0} кр" for uid, cr in totals) or "пусто")
    txt += f"\n\n_Дневной лимит: {DAILY_CREDIT_LIMIT} кр/юзера_"
    await msg.answer(txt)


@router.message(CommandStart())
async def cmd_start(msg: Message, state: FSMContext):
    if not is_allowed(msg.from_user.id):
        await deny(msg); return
    await state.clear()
    name = msg.from_user.first_name or "друг"
    await msg.answer(
        f"Привет, *{name}*! Я — MotionGen-агент.\n\n"
        "Делаю motion-control видео через Higgsfield (Kling 3.0). "
        "Ты даёшь фото-персонажа и видео-движение — я переношу движение с видео на фото "
        "и возвращаю готовый клип документом.\n\n"
        "Жми кнопку ниже 👇",
        reply_markup=MAIN_KB)


@router.callback_query(F.data == "help")
async def cb_help(cb: CallbackQuery):
    if not is_allowed(cb.from_user.id):
        return await cb.answer("🚫", show_alert=True)
    await cb.message.answer(
        "*Как сделать видео:*\n"
        "1. Жмёшь «🎬 Сделать видео»\n"
        "2. Присылаешь ФОТО (character — тот человек который будет «в кадре»)\n"
        "3. Присылаешь ВИДЕО (motion — движение/эмоция, которые надо перенести)\n"
        "4. Я показываю сколько кредитов уйдёт — ты подтверждаешь\n"
        "5. Жду ~10-15 мин, возвращаю готовый MP4 документом\n\n"
        "*Правила видео:* длина 3–30 сек, mp4/mov, с лицом или объектом в кадре.\n"
        "*Правила фото:* jpg/png, с лицом и телом видимым — именно их я «оживляю».")
    await cb.answer()


@router.callback_query(F.data.startswith("new_gen:"))
async def cb_new_gen(cb: CallbackQuery, state: FSMContext):
    if not is_allowed(cb.from_user.id):
        return await cb.answer("🚫", show_alert=True)
    if GEN_LOCK.locked():
        await cb.answer("Занят другой генерацией, подожди", show_alert=True); return
    backend = cb.data.split(":", 1)[1]
    if backend not in ("higgsfield", "vast"):
        await cb.answer("Неизвестный бэкенд", show_alert=True); return

    # Vast backend readiness guard: default workflow must exist
    if backend == "vast":
        wf_path = BASE / "workflows" / "default.json"
        if not wf_path.exists():
            await cb.message.answer(
                "⚠️ Для Vast.ai нужен workflow JSON.\n"
                "Положи его в `workflows/default.json` "
                "(Save API Format из ComfyUI) — и жми снова.",
                reply_markup=MAIN_KB)
            await cb.answer(); return

    await state.clear()
    await state.update_data(backend=backend)
    await state.set_state(GenState.awaiting_photo)
    nice = "Higgsfield (Kling 3.0)" if backend == "higgsfield" else "Vast.ai (Wan 2.2)"
    await cb.message.answer(
        f"*Бэкенд: {nice}*\n\n"
        "📸 Пришли *фото-референс* (character).\nТо лицо/человек которые будут «оживать».",
        reply_markup=CANCEL_KB)
    await cb.answer()


@router.callback_query(F.data == "cancel")
async def cb_cancel(cb: CallbackQuery, state: FSMContext):
    await state.clear()
    await cb.message.answer("Отменил. Ещё?", reply_markup=MAIN_KB)
    await cb.answer()


@router.message(GenState.awaiting_photo, F.photo | F.document)
async def got_photo(msg: Message, state: FSMContext):
    # Accept photos or documents with image mime
    if msg.photo:
        file_id = msg.photo[-1].file_id
        ext = ".jpg"
    else:
        doc = msg.document
        if not (doc.mime_type or "").startswith("image/"):
            await msg.reply("Это не картинка. Нужен JPG/PNG."); return
        file_id = doc.file_id
        ext = "." + (doc.mime_type.split("/")[-1].replace("jpeg", "jpg"))

    uid = msg.from_user.id
    dst = UPLOAD_DIR / f"u{uid}_photo_{int(time.time())}{ext}"
    await bot.download(file_id, destination=dst)
    await state.update_data(photo_path=str(dst))
    await state.set_state(GenState.awaiting_video)
    await msg.reply(
        f"✅ Фото принято ({dst.stat().st_size // 1024} КБ).\n\n"
        "🎥 Теперь пришли *видео-референс* (motion). 3–30 сек, mp4/mov.",
        reply_markup=CANCEL_KB)


@router.message(GenState.awaiting_photo)
async def bad_photo(msg: Message):
    await msg.reply("Жду ФОТО (как картинку или файл). Или нажми ❌.")


@router.message(GenState.awaiting_video, F.video | F.document | F.video_note)
async def got_video(msg: Message, state: FSMContext):
    if msg.video:
        file_id = msg.video.file_id
        ext = ".mp4"
    elif msg.video_note:
        file_id = msg.video_note.file_id
        ext = ".mp4"
    else:
        doc = msg.document
        mt = (doc.mime_type or "").lower()
        if "video" not in mt and not doc.file_name.lower().endswith((".mp4", ".mov", ".m4v")):
            await msg.reply("Это не видео. Нужен MP4/MOV."); return
        file_id = doc.file_id
        ext = os.path.splitext(doc.file_name or "video.mp4")[1] or ".mp4"

    uid = msg.from_user.id
    dst = UPLOAD_DIR / f"u{uid}_video_{int(time.time())}{ext}"
    wait = await msg.reply("⏳ Скачиваю видео и читаю параметры…")
    await bot.download(file_id, destination=dst)

    try:
        dur = engine.ffprobe_duration(str(dst))
    except Exception as e:
        await wait.edit_text(f"Не смог прочитать видео: `{e}`\nПришли другой файл.")
        return

    if dur < 3 or dur > 30:
        await wait.edit_text(
            f"Длина {dur:.1f} сек — не подходит.\n"
            "Нужно 3–30 сек. Пришли другой файл.")
        return

    cost = engine.estimate_cost(dur)
    used_today = credits_used_today(uid)
    remaining = DAILY_CREDIT_LIMIT - used_today

    if cost > remaining:
        await wait.edit_text(
            f"🚫 Дневной лимит кредитов почти исчерпан.\n\n"
            f"• На это видео нужно: *{cost}* кр\n"
            f"• Уже потрачено сегодня: *{used_today}* кр\n"
            f"• Осталось до лимита: *{remaining}* кр\n"
            f"• Лимит на юзера/день: *{DAILY_CREDIT_LIMIT}* кр\n\n"
            "Попробуй более короткое видео или подожди до завтра.")
        await state.clear()
        return

    await state.update_data(video_path=str(dst), duration=dur, cost=cost)
    await state.set_state(GenState.awaiting_confirm)

    import math
    await wait.edit_text(
        f"🎯 *Готов к запуску.*\n\n"
        f"• Длина видео: `{dur:.2f}` сек\n"
        f"• Стоимость: *~{cost} кредитов* (7 базовых + {(math.ceil(dur)-5)*3} за длину)\n"
        f"• Твой лимит сегодня: `{used_today}/{DAILY_CREDIT_LIMIT}` кр "
        f"(останется {remaining-cost} после этого видео)\n"
        f"• Время ожидания: ~10–15 минут\n"
        f"• Модель: Kling 3.0 Motion Control\n\n"
        f"Подтверди:",
        reply_markup=confirm_kb(cost))


@router.message(GenState.awaiting_video)
async def bad_video(msg: Message):
    await msg.reply("Жду ВИДЕО (mp4/mov, 3–30 сек). Или ❌.")


@router.callback_query(GenState.awaiting_confirm, F.data == "go")
async def cb_go(cb: CallbackQuery, state: FSMContext):
    if GEN_LOCK.locked():
        await cb.answer("Занят, подожди окончания текущей", show_alert=True); return
    data = await state.get_data()
    await state.clear()
    await cb.message.edit_text(f"🚀 Стартую генерацию…\n_Кредитов будет списано: ~{data['cost']}_")
    await cb.answer()

    chat_id = cb.message.chat.id
    status_msg = await bot.send_message(chat_id, "⏳ Поехали…")

    async def progress(text):
        try:
            await bot.edit_message_text(
                f"⏳ {text}", chat_id=chat_id, message_id=status_msg.message_id,
                parse_mode=None)
        except Exception as e:
            log.warning("progress edit failed: %s", e)

    async with GEN_LOCK:
        t0 = time.time()
        backend = data.get("backend", "higgsfield")
        try:
            await bot.send_chat_action(chat_id, ChatAction.RECORD_VIDEO)
            if backend == "vast":
                result = await engine.generate_via_vast(
                    data["photo_path"], data["video_path"],
                    work_dir=str(BASE / "output"),
                    provisioning_url=ENV.get("VAST_PROVISIONING_URL"),
                    on_progress=progress)
                result.setdefault("cost_credits", 0)
                result.setdefault("render_seconds", result.get("total_seconds", 0))
                result.setdefault("ref_video_duration", data["duration"])
                import engine as _e
                rw, rh = _e.ffprobe_size(result["final_path"])
                pw, ph = _e.ffprobe_size(data["photo_path"])
                result.update({"result_w": rw, "result_h": rh,
                               "photo_w": pw, "photo_h": ph})
            else:
                result = await engine.generate_motion(
                    data["photo_path"], data["video_path"],
                    work_dir=str(BASE / "output"),
                    on_progress=progress)
        except Exception as e:
            log.exception("generation failed")
            await bot.send_message(
                chat_id,
                f"❌ Ошибка: `{e}`\n\n```\n{traceback.format_exc()[-900:]}\n```")
            return

        elapsed = int(time.time() - t0)
        size_mb = os.path.getsize(result["final_path"]) / 1024 / 1024
        if backend == "higgsfield":
            record_usage(cb.from_user.id, result["cost_credits"])
            used_today_after = credits_used_today(cb.from_user.id)
            cost_line = f"• Кредитов потрачено: *{result['cost_credits']}*\n"
            daily_line = f"\n📊 Кредитов за сегодня: `{used_today_after}/{DAILY_CREDIT_LIMIT}`"
        else:
            used_today_after = credits_used_today(cb.from_user.id)
            cost_line = f"• Оценка стоимости: *~${result.get('cost_usd_estimate', 0):.3f}* (vast.ai, secondly billing)\n"
            daily_line = ""
        model_line = "Kling 3.0" if backend == "higgsfield" else "Wan 2.2 Fun Camera (vast.ai)"
        caption = (
            f"🎬 *Готово!* Motion Control — {model_line}\n\n"
            f"{cost_line}"
            f"• Время генерации: `{result['render_seconds']//60}:{result['render_seconds']%60:02d}`\n"
            f"• Общее время (с post-process): `{elapsed//60}:{elapsed%60:02d}`\n"
            f"• Длина результата: `{result['ref_video_duration']:.2f}` сек\n"
            f"• Разрешение: `{result['result_w']}×{result['result_h']}` (формат фото "
            f"{result['photo_w']}×{result['photo_h']})\n"
            f"• Размер файла: `{size_mb:.1f}` МБ\n\n"
            f"✅ Отправлен документом (не видео) — TG не пережмёт.\n"
            f"✅ Звук из твоего видео-референса сохранён.\n"
            f"✅ Aspect ratio подогнан под фото."
            f"{daily_line}")

        await bot.send_document(
            chat_id,
            FSInputFile(result["final_path"]),
            caption=caption,
            disable_content_type_detection=False)
        await bot.send_message(
            chat_id,
            "Ещё?", reply_markup=MAIN_KB)


# Catch-all for unexpected messages
@router.message()
async def fallback(msg: Message):
    if not is_allowed(msg.from_user.id):
        await deny(msg); return
    await msg.answer("Жми кнопку внизу 👇", reply_markup=MAIN_KB)


async def main():
    log.info("MotionGen bot starting. whitelist=%s", ALLOWED)
    me = await bot.get_me()
    log.info("logged in as @%s (id=%s)", me.username, me.id)
    await dp.start_polling(bot, allowed_updates=dp.resolve_used_update_types())


if __name__ == "__main__":
    asyncio.run(main())
