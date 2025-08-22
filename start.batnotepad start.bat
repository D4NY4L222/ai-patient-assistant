@echo off
REM === AI Patient Assistant Quick Start ===
cd backend
call .\venv\Scripts\activate
echo [INFO] Virtual environment activated.

REM Re-ingest FAQs only if you want (uncomment next line)
REM python ingest.py

echo [INFO] Starting server...
uvicorn main:app --reload --port 8000
pause
