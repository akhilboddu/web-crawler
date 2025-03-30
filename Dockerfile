FROM python:3.9-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY src/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ .

# Create directory for results
RUN mkdir -p results

# Set environment variables
ENV PORT=8016
ENV PYTHONUNBUFFERED=1

# Command to run the application
CMD uvicorn app:app --host 0.0.0.0 --port $PORT 