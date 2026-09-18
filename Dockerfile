FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY api ./api
COPY models ./models

EXPOSE 8000

CMD ["python", "api/api.py"]
