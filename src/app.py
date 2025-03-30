import os
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Import API router
from api import router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Chatwize Web Crawler API",
    description="REST API for crawling websites and preparing knowledge base data",
    version="1.0.0"
)

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Include the API router
app.include_router(router)

@app.get("/")
async def root():
    """Root endpoint with API info"""
    return {
        "name": "Chatwize Web Crawler API",
        "version": "1.0.0",
        "description": "REST API for crawling websites and preparing knowledge base data",
        "endpoints": {
            "POST /api/crawl": "Start a new crawl job",
            "GET /api/status/{job_id}": "Get status of a crawl job",
            "GET /api/results/{job_id}": "Get results of a completed crawl job",
            "GET /api/jobs": "List all crawl jobs"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8016))
    uvicorn.run("app:app", host="0.0.0.0", port=port) 