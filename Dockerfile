FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install ".[dev,llm]"

COPY tests ./tests
COPY data ./data
COPY results ./results

# Default: run the test suite. Examples:
#   docker run --rm judgelab judgelab benchmark --seeds 5
#   docker run --rm -e ANTHROPIC_API_KEY judgelab judgelab judge --backend anthropic ...
CMD ["python", "-m", "pytest", "-q", "-p", "no:cacheprovider"]
