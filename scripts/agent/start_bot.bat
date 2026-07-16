@echo off
rem WFEDS Telegram bot - double-click to start (keeps running in this window).
rem Needs TELEGRAM_BOT_TOKEN + an LLM key in the repo's .env.
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
python telegram_bot.py
pause
