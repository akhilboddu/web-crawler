from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
import logging
import argparse
from datetime import datetime
from typing import List, Dict, Any, Optional
import uvicorn
from crawler.crawler import WebCrawler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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

class CrawlResponse(BaseModel):
    base_url: str
    total_pages: int
    crawl_time: float
    start_time: str
    end_time: str
    pages: List[Dict[str, Any]]

@app.get("/")
async def root():
    """Verify that the API is running"""
    return {"status": "ok", "timestamp": datetime.now().isoformat()}

@app.post("/crawl", response_model=CrawlResponse)
async def crawl(request: CrawlRequest):
    """Crawl a website and return detailed analysis with page content"""
    try:
        logger.info(f"Starting crawl of {request.url}")
        
        crawler = WebCrawler()
        result = await crawler.crawl(str(request.url), request.max_pages, request.max_depth)
        
        logger.info(f"Completed crawl of {request.url}")
        return result
        
    except Exception as e:
        logger.error(f"Error during crawl: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def main():
    parser = argparse.ArgumentParser(description='Start the web crawler API server')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8001, help='Port to bind to')
    args = parser.parse_args()
    
    uvicorn.run("server:app", host=args.host, port=args.port, reload=True)

if __name__ == "__main__":
    main() 