FROM python:3.9-slim

# Set environment variables
ENV PORT=8016
ENV PYTHONUNBUFFERED=1

# Copy application code directly into the container
COPY src/ /app/

# Set working directory to where app.py is located
WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create results directory
RUN mkdir -p results

# Run app directly (no cd command needed)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8016"] 