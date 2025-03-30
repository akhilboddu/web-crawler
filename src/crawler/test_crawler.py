import asyncio
import aiohttp
import json
from typing import Dict
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

async def test_crawler(url: str, max_pages: int = 100, max_depth: int = 3) -> Dict:
    """
    Test the crawler by making a request to the crawler API.
    """
    api_base_url = "http://localhost:8000"
    
    # Wait for server to be ready
    server_ready = await wait_for_server(api_base_url)
    if not server_ready:
        logger.error("Server did not become available in time")
        return None
    
    # Prepare the request payload
    payload = {
        "url": url,
        "max_pages": max_pages,
        "max_depth": max_depth
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            logger.info(f"Starting crawl for URL: {url}")
            logger.info(f"Configuration: max_pages={max_pages}, max_depth={max_depth}")
            
            async with session.post(f"{api_base_url}/crawl", json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    
                    # Pretty print the results
                    logger.info("\n=== Crawl Results ===")
                    logger.info(f"Total pages found: {result['total_pages']}")
                    logger.info("\nDiscovered Pages:")
                    
                    # Sort pages by URL for better readability
                    sorted_pages = sorted(result['pages'], key=lambda x: x['url'])
                    
                    for page in sorted_pages:
                        logger.info(f"\nURL: {page['url']}")
                        logger.info(f"Title: {page['title']}")
                        logger.info(f"Content Type: {page['content_type']}")
                        logger.info("-" * 80)
                    
                    # Save results to a file
                    with open('crawl_results.json', 'w') as f:
                        json.dump(result, f, indent=2)
                    logger.info("\nResults have been saved to crawl_results.json")
                    
                    return result
                else:
                    error_text = await response.text()
                    logger.error(f"Error: {response.status} - {error_text}")
                    return None
                
    except aiohttp.ClientError as e:
        logger.error(f"Network error during crawl test: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error during crawl test: {str(e)}")
        return None

if __name__ == "__main__":
    # Test URLs - you can test with different websites
    test_urls = [
        "https://docs.crawl4ai.com",
        # Add more URLs to test here
    ]
    
    for url in test_urls:
        logger.info(f"\nTesting crawler with URL: {url}")
        # Run the test with smaller limits for testing
        asyncio.run(test_crawler(url, max_pages=10, max_depth=2))