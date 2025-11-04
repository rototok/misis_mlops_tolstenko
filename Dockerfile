FROM python:3.10-slim

WORKDIR /app

RUN python -m pip install --no-cache-dir pip setuptools wheel && \
    python -m pip install --no-cache-dir poetry==2.2.1

COPY . .

RUN poetry install --without dev --no-root --no-interaction --no-ansi

CMD ["poetry", "run", "pytest"]