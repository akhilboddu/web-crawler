import os
import json
from datetime import datetime
from typing import Dict, Any, Optional, List
from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from pydantic import BaseModel, HttpUrl
import logging

# Import our crawler
from crawler.crawler import WebCrawler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(
    prefix="/api",
    tags=["crawler"]
)

# Directory to store crawl results
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Models for API requests and responses
class CrawlRequest(BaseModel):
    url: HttpUrl
    max_pages: int = 100
    max_depth: int = 3
    
class CrawlResponse(BaseModel):
    job_id: str
    status: str
    url: str
    start_time: str
    
class CrawlStatus(BaseModel):
    job_id: str
    status: str
    url: str
    start_time: str
    end_time: Optional[str] = None
    total_pages: Optional[int] = None
    crawl_time: Optional[float] = None
    result_file: Optional[str] = None

# Store for active and completed crawls
active_crawls: Dict[str, Dict[str, Any]] = {}
completed_crawls: Dict[str, Dict[str, Any]] = {}

def generate_job_id() -> str:
    """Generate a unique job ID"""
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    return f"job_{timestamp}"

async def perform_crawl(job_id: str, url: str, max_pages: int, max_depth: int):
    """Background task to perform the crawl"""
    try:
        logger.info(f"Starting crawl job {job_id} for {url}")
        
        # Update job status
        active_crawls[job_id]["status"] = "crawling"
        
        # Initialize crawler
        crawler = WebCrawler()
        
        # Perform the crawl
        results = await crawler.crawl(url, max_pages, max_depth)
        
        # Generate a unique filename
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"{job_id}_{timestamp}.json"
        filepath = os.path.join(RESULTS_DIR, filename)
        
        # Save results to file
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Update job status and move to completed crawls
        active_crawls[job_id]["status"] = "completed"
        active_crawls[job_id]["end_time"] = datetime.now().isoformat()
        active_crawls[job_id]["total_pages"] = results.get("total_pages", 0)
        active_crawls[job_id]["crawl_time"] = results.get("crawl_time", 0)
        active_crawls[job_id]["result_file"] = filename
        
        # Move to completed crawls
        completed_crawls[job_id] = active_crawls[job_id].copy()
        del active_crawls[job_id]
        
        logger.info(f"Completed crawl job {job_id} for {url}")
        
    except Exception as e:
        logger.error(f"Error during crawl job {job_id}: {str(e)}")
        # Update job status to failed
        active_crawls[job_id]["status"] = "failed"
        active_crawls[job_id]["error"] = str(e)
        active_crawls[job_id]["end_time"] = datetime.now().isoformat()
        
        # Move to completed crawls
        completed_crawls[job_id] = active_crawls[job_id].copy()
        del active_crawls[job_id]

@router.post("/crawl", response_model=CrawlResponse)
async def start_crawl(request: CrawlRequest, background_tasks: BackgroundTasks):
    """Start a new crawl job"""
    job_id = generate_job_id()
    start_time = datetime.now().isoformat()
    
    # Create job entry
    job_info = {
        "job_id": job_id,
        "status": "starting",
        "url": str(request.url),
        "start_time": start_time,
        "max_pages": request.max_pages,
        "max_depth": request.max_depth,
    }
    
    # Add to active crawls
    active_crawls[job_id] = job_info
    
    # Start background task
    background_tasks.add_task(
        perform_crawl,
        job_id=job_id,
        url=str(request.url),
        max_pages=request.max_pages,
        max_depth=request.max_depth,
    )
    
    return CrawlResponse(
        job_id=job_id,
        status="starting",
        url=str(request.url),
        start_time=start_time
    )

@router.get("/status/{job_id}", response_model=CrawlStatus)
async def get_crawl_status(job_id: str):
    """Get the status of a crawl job"""
    if job_id in active_crawls:
        return CrawlStatus(**active_crawls[job_id])
    elif job_id in completed_crawls:
        return CrawlStatus(**completed_crawls[job_id])
    else:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

@router.get("/results/{job_id}")
async def get_crawl_results(job_id: str):
    """Get the results of a completed crawl job"""
    if job_id in completed_crawls:
        job_info = completed_crawls[job_id]
        if job_info.get("status") == "completed" and "result_file" in job_info:
            result_file = os.path.join(RESULTS_DIR, job_info["result_file"])
            try:
                with open(result_file, "r", encoding="utf-8") as f:
                    results = json.load(f)
                return results
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Error reading result file: {str(e)}"
                )
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Job {job_id} is not completed or has no results"
            )
    else:
        # Check active crawls as well
        if job_id in active_crawls:
            status = active_crawls[job_id].get("status", "unknown")
            raise HTTPException(
                status_code=400,
                detail=f"Job {job_id} is still in progress (status: {status})"
            )
        else:
            raise HTTPException(
                status_code=404, 
                detail=f"Job {job_id} not found"
            )

@router.get("/jobs")
async def list_jobs(status: Optional[str] = Query(None, description="Filter by status (active/completed/failed)")):
    """List all crawl jobs with optional status filter"""
    all_jobs = {}
    
    # Add active jobs
    for job_id, job_info in active_crawls.items():
        all_jobs[job_id] = job_info
    
    # Add completed jobs
    for job_id, job_info in completed_crawls.items():
        all_jobs[job_id] = job_info
    
    # Apply filter if specified
    if status:
        filtered_jobs = {}
        for job_id, job_info in all_jobs.items():
            job_status = job_info.get("status", "")
            if status == "active" and job_status in ["starting", "crawling"]:
                filtered_jobs[job_id] = job_info
            elif status == "completed" and job_status == "completed":
                filtered_jobs[job_id] = job_info
            elif status == "failed" and job_status == "failed":
                filtered_jobs[job_id] = job_info
        return filtered_jobs
    
    return all_jobs 