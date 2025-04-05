@echo off
REM start_local.bat
call venv\Scripts\activate
set API_KEY=your-local-test-key
uvicorn app.main:app --reload