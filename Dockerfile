FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends build-essential gcc && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY submission /app/submission
# The platform will call: python3 -m submission.main ...
