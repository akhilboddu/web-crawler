FROM python:3.9-slim

# Set work directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8016

# Install dependencies
COPY src/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY src/ .

# Create directories
RUN mkdir -p results

# Make port available
EXPOSE $PORT

# Run the application
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port $PORT"] 