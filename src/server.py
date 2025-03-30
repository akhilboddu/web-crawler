from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
import logging
import argparse
from datetime import datetime
import statistics
from typing import List, Dict, Any, Optional
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
    request: Dict[str, Any]
    timing: Dict[str, Any]
    statistics: Dict[str, Any]
    pages: List[Dict[str, Any]]

@app.get("/")
async def root():
    """Verify that the API is running"""
    return {"status": "ok", "timestamp": datetime.now().isoformat()}

@app.post("/crawl", response_model=CrawlResponse)
async def crawl(request: CrawlRequest):
    """Crawl a website and return detailed analysis"""
    try:
        logger.info(f"Starting crawl of {request.url}")
        start_time = datetime.now()
        
        # Create WebCrawler instance and call crawl method
        crawler = WebCrawler()
        result = await crawler.crawl(str(request.url), request.max_pages, request.max_depth)
        
        if not result:
            raise HTTPException(status_code=500, detail="Crawl failed")
        
        end_time = datetime.now()
        
        # Format response according to CrawlResponse model
        response = {
            "request": {
                "url": str(request.url),
                "max_pages": request.max_pages,
                "max_depth": request.max_depth
            },
            "timing": {
                "start_time": result.get("start_time", start_time.isoformat()),
                "end_time": result.get("end_time", end_time.isoformat()),
                "total_time": result.get("crawl_time", (end_time - start_time).total_seconds())
            },
            "statistics": {
                "total_pages": result.get("total_pages", 0),
                "crawl_time": result.get("crawl_time", 0)
            },
            "pages": result.get("pages", [])
        }
        
        logger.info(f"Completed crawl of {request.url}")
        return response
        
    except Exception as e:
        logger.error(f"Error during crawl: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def main():
    parser = argparse.ArgumentParser(description='Start the web crawler API server')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8001, help='Port to bind to')
    args = parser.parse_args()
    
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, reload=True)

if __name__ == "__main__":
    main() 