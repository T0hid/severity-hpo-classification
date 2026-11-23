FROM python:3.13-slim
# Liberation Sans for the figure text (see hpo_severity/style.py).
RUN apt-get update \
    && apt-get install -y --no-install-recommends fonts-liberation \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /work
COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install --no-cache-dir -r /tmp/requirements.txt
ENV MPLBACKEND=Agg PYTHONHASHSEED=0
