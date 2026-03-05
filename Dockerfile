FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements-dev.txt

COPY backend ./backend
COPY frontend ./frontend
COPY mcp_servers ./mcp_servers
COPY scripts ./scripts
COPY deploy ./deploy
COPY terraform ./terraform
COPY tests ./tests
COPY README.md ./
COPY AGENT.md ./
COPY NOTICE.md ./
COPY .gitignore ./
RUN mkdir -p /app/reports
COPY pytest.ini ./

EXPOSE 8000

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
