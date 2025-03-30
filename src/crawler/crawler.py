import aiohttp
import asyncio
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import json
import ssl
import logging
from datetime import datetime
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class WebCrawler:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        self.visited_urls = set()
        self.discovered_pages = []
        self.error_count = 0
        self.max_errors = 5

    def clean_url(self, url: str) -> str:
        url = url.split('#')[0]
        if url.endswith('/'):
            url = url[:-1]
        return url

    async def is_same_domain(self, base_url: str, url: str) -> bool:
        try:
            base_domain = urlparse(base_url).netloc.lower()
            url_domain = urlparse(url).netloc.lower()
            base_domain = base_domain.replace('www.', '')
            url_domain = url_domain.replace('www.', '')
            return base_domain == url_domain
        except Exception as e:
            logger.error(f"Error parsing domain: {str(e)}")
            return False

    async def get_page_content(self, session: aiohttp.ClientSession, url: str) -> dict:
        try:
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE

            logger.info(f"Fetching: {url}")
            async with session.get(url, headers=self.headers, ssl=ssl_context) as response:
                if response.status == 200:
                    content_type = response.headers.get('content-type', '')
                    if 'text/html' in content_type.lower():
                        html = await response.text()
                        soup = BeautifulSoup(html, 'html.parser')
                        title = soup.title.string if soup.title else 'No title'
                        title = title.strip()

                        links = []
                        for link in soup.find_all('a', href=True):
                            href = link['href']
                            if href and not href.startswith(('#', 'javascript:', 'mailto:', 'tel:')):
                                clean_href = self.clean_url(urljoin(url, href))
                                if clean_href:
                                    links.append(clean_href)

                        return {
                            'success': True,
                            'title': title,
                            'links': links,
                            'content_type': content_type
                        }
                    else:
                        return {
                            'success': True,
                            'title': url,
                            'links': [],
                            'content_type': content_type
                        }
                else:
                    return {'success': False, 'error': f"HTTP {response.status}"}
        except Exception as e:
            logger.error(f"Error fetching {url}: {str(e)}")
            return {'success': False, 'error': str(e)}

    async def crawl_page(self, session: aiohttp.ClientSession, url: str, depth: int, max_depth: int):
        if depth > max_depth or len(self.discovered_pages) >= self.max_pages:
            return

        if self.error_count >= self.max_errors:
            logger.error("Too many errors encountered, stopping crawl")
            return

        clean_url = self.clean_url(url)
        if clean_url in self.visited_urls:
            return

        self.visited_urls.add(clean_url)

        try:
            result = await self.get_page_content(session, clean_url)
            if result['success']:
                page_info = {
                    "url": clean_url,
                    "title": result['title'],
                    "content_type": result['content_type'],
                    "depth": str(depth)
                }
                self.discovered_pages.append(page_info)
                logger.info(f"Crawled: {clean_url} (depth: {depth})")

                if depth < max_depth:
                    for link in result['links']:
                        if await self.is_same_domain(url, link):
                            await self.crawl_page(session, link, depth + 1, max_depth)
            else:
                self.error_count += 1
                logger.warning(f"Failed to crawl {clean_url}: {result.get('error')}")
        except Exception as e:
            self.error_count += 1
            logger.error(f"Error crawling {clean_url}: {str(e)}")

    async def crawl(self, url: str, max_pages: int = 10, max_depth: int = 2) -> dict:
        self.visited_urls.clear()
        self.discovered_pages.clear()
        self.error_count = 0
        self.max_pages = max_pages

        start_time = datetime.now()
        try:
            timeout = aiohttp.ClientTimeout(total=30)
            connector = aiohttp.TCPConnector(ssl=False)
            async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
                await self.crawl_page(session, url, 0, max_depth)
        except Exception as e:
            logger.error(f"Error in crawl process: {str(e)}")

        end_time = datetime.now()
        crawl_time = (end_time - start_time).total_seconds()

        result = {
            "pages": self.discovered_pages,
            "total_pages": len(self.discovered_pages),
            "crawl_time": crawl_time,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat()
        }

        try:
            with open('crawl_results.json', 'w') as f:
                json.dump(result, f, indent=2)
            logger.info("Results saved to crawl_results.json")
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")

        return result

async def main():
    url = "https://www.zaio.io"
    max_pages = 10
    max_depth = 2

    logger.info(f"Starting crawl for {url}")
    logger.info(f"Configuration: max_pages={max_pages}, max_depth={max_depth}")

    crawler = WebCrawler()
    result = await crawler.crawl(url, max_pages, max_depth)

    print("\nCrawl Results:")
    print(f"Total pages crawled: {result['total_pages']}")
    print(f"Crawl time: {result['crawl_time']:.2f} seconds")
    print("\nPages discovered:")
    for page in result['pages']:
        print(f"- {page['url']} (depth: {page['depth']})")

if __name__ == "__main__":
    asyncio.run(main())