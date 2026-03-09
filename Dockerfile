FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY config_gen.py .
COPY group_params.json .
COPY router.py .

EXPOSE 11434

CMD ["uvicorn", "router:app", "--host", "0.0.0.0", "--port", "11434", "--log-level", "info"]
