FROM python:3.9-slim

WORKDIR /app

# Copy requirements and install
COPY src/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ /app/

# Set environment variables
ENV PORT=8016

# Run the application
CMD bash -c "cd /app && uvicorn app:app --host 0.0.0.0 --port ${PORT}" 