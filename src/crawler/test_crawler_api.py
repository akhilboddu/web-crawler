import asyncio
import aiohttp
import json
from typing import Dict
import logging
import time
from datetime import datetime
import pytest

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

API_BASE_URL = "http://localhost:8002"

async def wait_for_server(url: str, max_retries: int = 5, delay: int = 2) -> bool:
    """Wait for the server to become available."""
    for i in range(max_retries):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{url}/health") as response:
                    if response.status == 200:
                        return True
        except aiohttp.ClientError:
            logger.info(f"Server not ready, attempt {i+1}/{max_retries}. Waiting {delay} seconds...")
            await asyncio.sleep(delay)
    return False

async def monitor_progress(session: aiohttp.ClientSession, task_id: str) -> bool:
    """Monitor the progress of a crawl task."""
    while True:
        try:
            async with session.get(f"{API_BASE_URL}/status/{task_id}") as response:
                if response.status == 200:
                    status = await response.json()
                    logger.info(f"Status: {status['status']}")
                    logger.info(f"Pages crawled: {status.get('pages_crawled', 0)}/{status.get('total_pages', '?')}")
                    logger.info(f"Phase: {status.get('phase', 'N/A')}")
                    logger.info(f"URLs in Queue: {status.get('urls_in_queue', 'N/A')}")
                    logger.info(f"Current URL: {status.get('current_url', 'unknown')}")
                    errors = status.get('errors', [])
                    if errors:
                        logger.warning(f"Errors reported ({len(errors)}):")
                        for err in errors[-3:]:
                            logger.warning(f"  - {err}")
                    
                    if status["status"] in ["completed", "failed"]:
                        return status["status"] == "completed"
                    
                    await asyncio.sleep(2)  # Wait before next check
                else:
                    logger.error(f"Error checking progress: {response.status}")
                    return False
        except Exception as e:
            logger.error(f"Error monitoring progress: {str(e)}")
            return False

@pytest.fixture
async def server_ready():
    """Fixture to ensure server is ready before running tests."""
    is_ready = await wait_for_server(API_BASE_URL)
    if not is_ready:
        pytest.skip("Server is not available")
    return is_ready

@pytest.mark.asyncio
async def test_crawler_health(server_ready):
    """Test the health endpoint."""
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{API_BASE_URL}/health") as response:
            assert response.status == 200
            data = await response.json()
            assert data["status"] == "ok"

@pytest.mark.asyncio
async def test_crawler_workflow(server_ready):
    """Test the complete crawler workflow."""
    url = "https://www.siza.io"
    max_pages = 3  # Reduced for faster testing
    max_depth = 1  # Add max_depth parameter
    
    async with aiohttp.ClientSession() as session:
        # Start crawl
        payload = {
            "url": url,
            "max_pages": max_pages,
            "max_depth": max_depth
        }
        logger.info(f"Starting crawl with payload: {payload}")
        async with session.post(
            f"{API_BASE_URL}/crawl",
            json=payload
        ) as response:
            assert response.status == 200
            data = await response.json()
            assert "task_id" in data
            task_id = data["task_id"]
            logger.info(f"Crawl task started with ID: {task_id}")
        
        # Monitor progress
        success = await monitor_progress(session, task_id)
        assert success, "Crawl task did not complete successfully"
        
        # Get final status
        async with session.get(f"{API_BASE_URL}/status/{task_id}") as response:
            assert response.status == 200
            status = await response.json()
            assert status["status"] == "completed"
            assert "results_file" in status
            logger.info(f"Results saved to: {status['results_file']}")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])