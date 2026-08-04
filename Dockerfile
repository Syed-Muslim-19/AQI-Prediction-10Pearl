FROM python:3.12-slim

WORKDIR /app

# Install system dependencies needed by some packages (if any)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt ./
RUN python -m pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . /app

EXPOSE 8000

# Default command runs the FastAPI app. To run the Streamlit dashboard, override the command.
CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8000"]
