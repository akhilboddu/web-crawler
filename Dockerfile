FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY src/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./

# Create required directories
RUN mkdir -p results

# Set environment variables
ENV PORT=8016
ENV PYTHONPATH=/app

# Expose port
EXPOSE 8016

# Command to run the application
CMD ["gunicorn", "app:app", "--workers", "4", "--worker-class", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8016", "--timeout", "120"] 