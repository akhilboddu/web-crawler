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

# Make the start script executable
RUN chmod +x start.sh

# Run the application using the shell script
CMD ["./start.sh"] 