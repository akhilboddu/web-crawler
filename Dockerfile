FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8016

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the source code
COPY src/ .

# Create results directory
RUN mkdir -p results

# Expose the port
EXPOSE 8016

# Run the application
CMD uvicorn server:app --host 0.0.0.0 --port ${PORT} 