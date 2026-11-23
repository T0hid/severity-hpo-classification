FROM python:3.13-slim
# Fonts used by the figures (see hpo_severity/style.py). Without them matplotlib uses DejaVu Sans.
RUN apt-get update \
    && apt-get install -y --no-install-recommends fonts-liberation fonts-inter \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /work
COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install --no-cache-dir -r /tmp/requirements.txt
ENV MPLBACKEND=Agg PYTHONHASHSEED=0
