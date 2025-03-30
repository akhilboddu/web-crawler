# Chatwize Web Crawler

A web crawler API for extracting content from websites, built with FastAPI and asyncio.

[![Deploy to Railway](https://railway.app/button.svg)](https://railway.app)

## Features

- Fast and efficient web crawling using async I/O
- Configurable crawl depth and page limits
- Background job processing for long-running crawls
- REST API with comprehensive endpoint documentation

## Requirements

- Python 3.9+
- Docker (for containerized deployment)

## Quick Start

### Local Development

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/chatwize-web-crawler.git
   cd chatwize-web-crawler
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use: venv\Scripts\activate
   pip install -r src/requirements.txt
   ```

3. Run the application:
   ```bash
   cd src
   uvicorn app:app --reload
   ```

4. Access the API documentation at [http://localhost:8016/docs](http://localhost:8016/docs)

### Using Docker

1. Build the Docker image:
   ```bash
   docker build -t chatwize-crawler .
   ```

2. Run the container:
   ```bash
   docker run -p 8016:8016 chatwize-crawler
   ```

3. Access the API at [http://localhost:8016](http://localhost:8016)

## API Endpoints

- `GET /` - API information
- `GET /health` - Health check endpoint
- `POST /api/crawl` - Start a new crawl job
- `GET /api/status/{job_id}` - Get status of a crawl job
- `GET /api/results/{job_id}` - Get results of a completed crawl job
- `GET /api/jobs` - List all crawl jobs

## Example Usage

### Starting a Crawl

```bash
curl -X POST http://localhost:8016/api/crawl \
  -H "Content-Type: application/json" \
  -d '{"url": "https://www.zaio.io", "max_depth": 2, "max_pages": 10}'
```

Response:
```json
{
  "job_id": "job_20231115123456",
  "status": "starting",
  "url": "https://www.zaio.io",
  "start_time": "2023-11-15T12:34:56.789012"
}
```

### Checking Status

```bash
curl http://localhost:8016/api/status/job_20231115123456
```

### Retrieving Results

```bash
curl http://localhost:8016/api/results/job_20231115123456
```

## Deployment

This project is configured for deployment on Railway. Push to the `main` branch to trigger automatic deployment via GitHub Actions.

### Railway Setup

1. Create a Railway account and project
2. Add your Railway API token as a GitHub secret named `RAILWAY_TOKEN`
3. Push to the main branch to trigger deployment

## Environment Variables

- `PORT` - Port to run the server on (default: 8016)

## License

MIT