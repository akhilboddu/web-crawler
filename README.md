# Chatwize Web Crawler API

A reliable web crawler REST API designed for building knowledge bases.

## Features

- **Multi-page crawling**: Crawl multiple pages within a domain with depth control
- **Content extraction**: Extract clean text, HTML, and metadata from pages
- **Asynchronous processing**: Run crawl jobs in the background
- **Job management**: Start, monitor, and retrieve crawl jobs
- **Knowledge base ready**: Output is formatted for direct knowledge base ingestion

## Deployment

### Render Deployment

1. Sign up for a [Render account](https://render.com/)
2. Create a new Web Service and connect your GitHub repository
3. Configure as follows:
   - Environment: Docker
   - Branch: main
   - Plan: Free
   
### Continuous Deployment

For automated deployments via GitHub Actions:

1. In your Render dashboard, get your Service ID and create an API Key
2. Add these secrets to your GitHub repository:
   - `RENDER_SERVICE_ID`: Your Render service ID
   - `RENDER_API_KEY`: Your Render API key

## Local Development

### Docker

Build and run using Docker:

```bash
docker build -t web-crawler .
docker run -p 8016:8016 web-crawler
```

### Docker Compose

```bash
docker-compose up
```

## Getting Started

### Prerequisites

- Python 3.9+
- `pip` package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/chatwize-web-crawler.git
cd chatwize-web-crawler
```

2. Create and activate a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the API Server

```bash
cd src
python api.py
```

The API server will start on `http://localhost:8016` with auto-reload enabled.

## API Endpoints

### Start a new crawl

```
POST /crawl
```

Request body:
```json
{
  "url": "https://example.com",
  "max_pages": 100,
  "max_depth": 3,
  "delay": 1.0
}
```

Response:
```json
{
  "job_id": "job_20250401123456",
  "status": "starting",
  "url": "https://example.com",
  "start_time": "2025-04-01T12:34:56.789012"
}
```

### Check crawl status

```
GET /status/{job_id}
```

Response:
```json
{
  "job_id": "job_20250401123456",
  "status": "completed",
  "url": "https://example.com",
  "start_time": "2025-04-01T12:34:56.789012",
  "end_time": "2025-04-01T12:35:30.123456",
  "total_pages": 10,
  "crawl_time": 33.33,
  "result_file": "job_20250401123456_20250401123530.json"
}
```

### Get crawl results

```
GET /results/{job_id}
```

Returns the full JSON crawl results.

### List all jobs

```
GET /jobs
```

Returns a list of all crawl jobs.

### List jobs by status

```
GET /jobs?status=completed
```

Returns jobs filtered by status (active, completed, failed).

### Cancel a job

```
POST /cancel/{job_id}
```

Cancels an active crawl job.

### Delete a job

```
DELETE /jobs/{job_id}
```

Deletes a completed job and its results.

## Output Format

The crawler produces a structured JSON output with the following format:

```json
{
  "source": "https://example.com",
  "domain": "example.com",
  "start_time": "2025-04-01T12:34:56.789012",
  "end_time": "2025-04-01T12:35:30.123456",
  "crawl_time": 33.33,
  "total_pages": 10,
  "max_depth": 3,
  "pages": [
    {
      "url": "https://example.com",
      "title": "Example Website",
      "content": "Example Website\n\nThis is a sample description\n\nWelcome to our website...",
      "html": "<!DOCTYPE html><html>...</html>",
      "text_blocks": ["Welcome to our website", "About us", "..."],
      "meta_description": "This is a sample description",
      "links": [
        {
          "url": "https://example.com/about",
          "text": "About"
        },
        ...
      ]
    },
    ...
  ]
}
```

## Building a Knowledge Base

The output from this crawler is specifically designed for knowledge base ingestion:

1. **Content extraction**: Clean text is extracted from the HTML and organized into blocks
2. **Metadata preservation**: Titles, descriptions, and URLs are preserved
3. **Structure maintained**: The relationship between pages is maintained through links
4. **Full-text search ready**: The content field provides a consolidated text representation

## License

MIT