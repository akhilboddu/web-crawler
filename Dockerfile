FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8016

# Create and set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY src/requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ /app/

# Create results directory
RUN mkdir -p results

# Expose the port
EXPOSE ${PORT}

# Run the application
CMD uvicorn app:app --host 0.0.0.0 --port ${PORT} 