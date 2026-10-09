"""
Gold Telegram Signal Bot — educational prototype.
Data source: Yahoo Finance COMEX gold futures (GC=F), NOT broker-specific spot XAUUSD.
Signals are illustrative, not financial advice. No trade execution.
"""
import os
import json
import logging
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import yfinance as yf
from telegram import Update
from telegram.ext import (
    Application, CommandHandler, ContextTypes
)

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
POLL_SECONDS = int(os.getenv("POLL_SECONDS", "60"))
SYMBOL = os.getenv("YAHOO_SYMBOL", "GC=F")
SUBSCRIBERS_FILE = Path(os.getenv("SUBSCRIBERS_FILE", "subscribers.json"))
STATE_FILE = Path(os.getenv("STATE_FILE", "state.json"))

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    level=logging.INFO,
)
log = logging.getLogger("gold_signal_bot")


def read_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return default


def write_json(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def get_subscribers():
    return set(str(x) for x in read_json(SUBSCRIBERS_FILE, []))


def save_subscribers(subs):
    write_json(SUBSCRIBERS_FILE, sorted(subs))


def get_state():
    return read_json(STATE_FILE, {"last_signal_key": None, "last_checked_candle": None})


def save_state(state):
    write_json(STATE_FILE, state)


def fetch_candles():
    # Yahoo's intraday history is limited and may be delayed/unavailable.
    df = yf.download(
        SYMBOL, period="5d", interval="5m",
        auto_adjust=False, progress=False, threads=False
    )
    if df is None or df.empty:
        raise RuntimeError("No candles returned by Yahoo Finance.")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    needed = ["Open", "High", "Low", "Close"]
    if any(c not in df.columns for c in needed):
        raise RuntimeError(f"Unexpected data columns: {list(df.columns)}")
    df = df[needed].dropna().copy()
    # Ignore the currently forming 5-minute candle.
    if len(df) < 220:
        raise RuntimeError(f"Not enough candles yet ({len(df)}); wait for more data.")
    return df.iloc[:-1].copy()


def analyze():
    df = fetch_candles()
    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    df["ema50"] = close.ewm(span=50, adjust=False).mean()
    df["ema200"] = close.ewm(span=200, adjust=False).mean()

    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / loss.replace(0, float("nan"))
    df["rsi"] = 100 - (100 / (1 + rs))

    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    df["atr"] = tr.ewm(alpha=1/14, adjust=False).mean()

    # Breakout range excludes the signal candle.
    df["prior_high"] = high.shift(1).rolling(10).max()
    df["prior_low"] = low.shift(1).rolling(10).min()

    row = df.iloc[-1]
    candle_time = df.index[-1]
    if getattr(candle_time, "tzinfo", None) is None:
        candle_time = candle_time.tz_localize("UTC")
    candle_time = candle_time.to_pydatetime().astimezone(timezone.utc)

    vals = [row[c] for c in ["ema50", "ema200", "rsi", "atr", "prior_high", "prior_low"]]
    if any(pd.isna(v) for v in vals) or row["atr"] <= 0:
        return {"status": "waiting", "time": candle_time.isoformat()}

    rng = float(row["High"] - row["Low"])
    close_price = float(row["Close"])
    atr = float(row["atr"])
    bullish = row["ema50"] > row["ema200"] and close_price > row["ema50"]
    bearish = row["ema50"] < row["ema200"] and close_price < row["ema50"]
    candle_ok = rng <= 2.0 * atr

    buy = bullish and close_price > row["prior_high"] and 52 <= row["rsi"] <= 68 and candle_ok
    sell = bearish and close_price < row["prior_low"] and 32 <= row["rsi"] <= 48 and candle_ok

    result = {
        "status": "no_signal",
        "time": candle_time.isoformat(),
        "close": close_price,
        "rsi": float(row["rsi"]),
        "atr": atr,
        "symbol": SYMBOL,
    }
    if buy or sell:
        side = "BUY" if buy else "SELL"
        entry = close_price
        sl_dist = 1.5 * atr
        sl = entry - sl_dist if side == "BUY" else entry + sl_dist
        tp = entry + 2 * sl_dist if side == "BUY" else entry - 2 * sl_dist
        result.update({
            "status": "signal", "side": side, "entry": entry,
            "sl": sl, "tp": tp, "rr": 2.0,
            "key": f"{candle_time.isoformat()}:{side}",
        })
    return result


def fmt_price(value):
    return f"{value:,.2f}"


def signal_message(s):
    icon = "🟢" if s["side"] == "BUY" else "🔴"
    return (
        f"{icon} *Gold signal — {s['side']}*\n"
        f"Symbol: `{s['symbol']}` (GC=F futures proxy)\n"
        f"Timeframe: `M5` | Closed candle: `{s['time']}`\n\n"
        f"Entry reference: `{fmt_price(s['entry'])}`\n"
        f"Stop loss (ATR): `{fmt_price(s['sl'])}`\n"
        f"Take profit (RR 1:2): `{fmt_price(s['tp'])}`\n"
        f"RSI(14): `{s['rsi']:.1f}`\n\n"
        "⚠️ Educational signal only. GC=F is gold futures, not your broker's exact XAUUSD quote. "
        "Confirm current price, spread, contract specs and execution before considering any trade. "
        "No win-rate or profit is guaranteed."
    )


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = str(update.effective_chat.id)
    subs = get_subscribers()
    subs.add(chat_id)
    save_subscribers(subs)
    await update.message.reply_text(
        "سلام! ربات آزمایشی سیگنال طلا فعال شد.\n\n"
        "دستورها:\n"
        "/status — وضعیت داده و تحلیل\n"
        "/latest — آخرین وضعیت سیگنال\n"
        "/stop — توقف اعلان‌ها\n"
        "/help — راهنما\n\n"
        "توجه: داده از GC=F (قرارداد آتی طلا) است و لزوماً با XAUUSD بروکر یکی نیست. "
        "وین‌ریت تضمین نمی‌شود."
    )


async def stop(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = str(update.effective_chat.id)
    subs = get_subscribers()
    subs.discard(chat_id)
    save_subscribers(subs)
    await update.message.reply_text("اعلان‌های این چت متوقف شد. برای فعال‌سازی دوباره /start را بزن.")


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "/start — فعال‌سازی اعلان‌ها\n"
        "/status — دریافت و تحلیل داده فعلی\n"
        "/latest — وضعیت آخرین کندل بسته‌شده\n"
        "/stop — لغو اعلان‌ها\n\n"
        "این ربات معامله‌ای اجرا نمی‌کند و تضمین سود یا وین‌ریت ندارد."
    )


async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        s = analyze()
        if s["status"] == "signal":
            await update.message.reply_text(signal_message(s), parse_mode="Markdown")
        elif s["status"] == "no_signal":
            await update.message.reply_text(
                f"فعلاً سیگنال معتبر نیست.\n"
                f"نماد داده: {SYMBOL} (futures proxy)\n"
                f"آخرین کندل بسته‌شده: {s['time']}\n"
                f"قیمت مرجع: {fmt_price(s['close'])}\nRSI: {s['rsi']:.1f}"
            )
        else:
            await update.message.reply_text("داده کافی برای تحلیل موجود نیست؛ کمی بعد دوباره امتحان کن.")
    except Exception as e:
        log.exception("status analysis failed")
        await update.message.reply_text(
            "دریافت داده فعلاً ناموفق بود. ممکن است منبع داده در دسترس نباشد یا تاریخچه کافی نداشته باشد.\n"
            f"جزئیات: {str(e)[:250]}"
        )


async def latest(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await status(update, context)


async def monitor(context: ContextTypes.DEFAULT_TYPE):
    subs = get_subscribers()
    if not subs:
        return
    try:
        s = analyze()
        state = get_state()
        state["last_checked_candle"] = s.get("time")
        if s.get("status") == "signal" and s.get("key") != state.get("last_signal_key"):
            message = signal_message(s)
            for chat_id in list(subs):
                try:
                    await context.bot.send_message(chat_id=int(chat_id), text=message, parse_mode="Markdown")
                except Exception:
                    log.exception("Failed to send to chat %s", chat_id)
            state["last_signal_key"] = s["key"]
        save_state(state)
    except Exception:
        log.exception("monitor failed")


def main():
    if not TOKEN:
        raise SystemExit(
            "Set TELEGRAM_BOT_TOKEN environment variable first. "
            "Do not put your token in public code or send it to anyone."
        )
    app = Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("stop", stop))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("status", status))
    app.add_handler(CommandHandler("latest", latest))
    app.job_queue.run_repeating(monitor, interval=POLL_SECONDS, first=15)
    log.info("Bot started. Poll interval=%s sec; symbol=%s", POLL_SECONDS, SYMBOL)
    app.run_polling()


if __name__ == "__main__":
    main()
