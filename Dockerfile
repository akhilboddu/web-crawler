FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8016

# Copy requirements and install
COPY src/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Copy all source code to /app/src
COPY src/ /app/src/

# Set the working directory to /app/src where the app.py is located
WORKDIR /app/src

# Create results directory
RUN mkdir -p results

# Run the application directly from the working directory
CMD uvicorn app:app --host 0.0.0.0 --port ${PORT} 