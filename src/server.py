from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, HttpUrl, field_validator
import logging
import argparse
from datetime import datetime
from typing import List, Dict, Any, Optional
import uvicorn
from crawler.crawler import WebCrawler
from crawler.state import crawl_progress
import asyncio
from urllib.parse import urlparse
import json
import time
import os
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define results directory
RESULTS_DIR = Path("/app/results")

app = FastAPI(title="Web Crawler API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class CrawlRequest(BaseModel):
    url: HttpUrl
    max_pages: int = 50
    max_depth: int = 3
    concurrency: int = 5 # Default concurrency for workers
    timeout: Optional[int] = 300  # timeout in seconds, default 5 minutes

    @field_validator('max_pages')
    @classmethod
    def validate_max_pages(cls, v):
        if v <= 0 or v > 100:
            raise ValueError('max_pages must be between 1 and 100')
        return v

    @field_validator('max_depth')
    @classmethod
    def validate_max_depth(cls, v):
        if v <= 0 or v > 5:
            raise ValueError('max_depth must be between 1 and 5')
        return v

    @field_validator('concurrency')
    @classmethod
    def validate_concurrency(cls, v):
        if v <= 0 or v > 10: # Limit concurrency
            raise ValueError('concurrency must be between 1 and 10')
        return v

    @field_validator('timeout')
    @classmethod
    def validate_timeout(cls, v):
        if v is not None and (v < 30 or v > 600):
            raise ValueError('timeout must be between 30 and 600 seconds')
        return v

class CrawlResponse(BaseModel):
    task_id: str
    status: str
    message: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class CrawlProgressResponse(BaseModel):
    task_id: str
    status: str
    pages_crawled: int
    total_pages: int
    current_url: Optional[str] = None
    elapsed_time: float
    estimated_time_remaining: Optional[float] = None

@app.get("/")
async def root():
    """Root endpoint to verify API is running"""
    return {
        "status": "ok",
        "message": "Web Crawler API is running",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok"}

@app.post("/crawl")
async def crawl_site(request: CrawlRequest):
    """Start a new crawl task"""
    # Convert HttpUrl to string and create task_id
    url_str = str(request.url)
    parsed_url = urlparse(url_str)
    task_id = f"{parsed_url.netloc}_{int(time.time())}"
    
    # Initialize progress tracking
    crawl_progress[task_id] = {
        "status": "initializing",
        "pages_crawled": 0,
        "total_pages": request.max_pages,
        "max_depth": request.max_depth,
        "concurrency": request.concurrency,
        "current_url": url_str,
        "start_time": datetime.now().isoformat(),
        "last_update": datetime.now().isoformat(),
        "errors": [] # Initialize errors list
    }
    
    # Start crawl in background task
    asyncio.create_task(run_crawl(task_id, request))
    
    return {"task_id": task_id}

@app.get("/status/{task_id}")
async def get_status(task_id: str):
    """Get the status of a crawl task"""
    if task_id not in crawl_progress:
        raise HTTPException(status_code=404, detail="Task not found")
    return crawl_progress[task_id]

@app.get("/results/")
async def list_results():
    """List available result files"""
    if not RESULTS_DIR.is_dir():
        logger.warning(f"Results directory not found: {RESULTS_DIR}")
        return [] # Return empty list if dir doesn't exist
    
    try:
        result_files = [f for f in os.listdir(RESULTS_DIR) if f.endswith('.json') and os.path.isfile(RESULTS_DIR / f)]
        return sorted(result_files) # Return sorted list
    except Exception as e:
        logger.error(f"Error listing results directory {RESULTS_DIR}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Could not list result files.")

@app.get("/results/{filename}")
async def get_result_file(filename: str):
    """Download a specific result file"""
    if not RESULTS_DIR.is_dir():
         raise HTTPException(status_code=404, detail=f"Results directory not found.")

    try:
        # Security: Ensure filename is just a filename and prevent path traversal
        if "/" in filename or ".." in filename:
            raise HTTPException(status_code=400, detail="Invalid filename.")
            
        file_path = RESULTS_DIR / filename

        # Security: Double check it's a file and exists within the results dir
        if not file_path.is_file() or not file_path.resolve().is_relative_to(RESULTS_DIR.resolve()):
             raise HTTPException(status_code=404, detail=f"Result file not found or invalid: {filename}")

        return FileResponse(path=file_path, filename=filename, media_type='application/json')

    except HTTPException as http_exc: # Re-raise HTTP exceptions
        raise http_exc
    except Exception as e:
        logger.error(f"Error accessing result file {filename}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Could not access result file: {filename}")

async def run_crawl(task_id: str, request: CrawlRequest):
    """Run the crawl task"""
    try:
        # Initialize crawler with task_id for progress tracking
        crawler = WebCrawler(task_id=task_id)
        
        # Run the crawl, passing all relevant parameters
        result = await crawler.crawl(
            str(request.url),
            max_pages=request.max_pages,
            max_depth=request.max_depth, # Pass max_depth
            concurrency=request.concurrency # Pass concurrency
        )
        
        # Update progress with completion status
        results_filepath = result.get("crawl_metadata", {}).get("results_file")
        crawl_progress[task_id].update({
            "status": "completed",
            "results_file": results_filepath,
            "end_time": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error during crawl: {str(e)}", exc_info=True)
        # Update progress with error status
        crawl_progress[task_id].update({
            "status": "failed",
            "error": str(e),
            "end_time": datetime.now().isoformat()
        })

def main():
    parser = argparse.ArgumentParser(description='Start the web crawler API server')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8002, help='Port to bind to')
    args = parser.parse_args()
    
    uvicorn.run("server:app", host=args.host, port=args.port, reload=True)

if __name__ == "__main__":
    main() 