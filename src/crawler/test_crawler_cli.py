import asyncio
import argparse
import logging
from crawler import WebCrawler
from urllib.parse import urlparse, urljoin

# Configure logging with more detail
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def normalize_url(url: str) -> str:
    """Normalize URL by adding https:// if protocol is missing"""
    if not url.startswith(('http://', 'https://')):
        return f'https://{url}'
    return url

async def run_crawler(url: str, max_pages: int = 100, max_depth: int = 3):
    """
    Run the crawler directly using the WebCrawler class.
    """
    try:
        # Normalize URL
        normalized_url = normalize_url(url)
        logger.info(f"Starting crawl for URL: {normalized_url}")
        logger.info(f"Configuration: max_pages={max_pages}, max_depth={max_depth}")
        
        # Initialize and run crawler
        crawler = WebCrawler()
        result = await crawler.crawl(normalized_url, max_pages=max_pages, max_depth=max_depth)
        
        # Print detailed results
        logger.info("\n=== Crawl Results ===")
        logger.info(f"Total pages crawled: {result['total_pages']}")
        logger.info(f"Crawl time: {result['crawl_time']:.2f} seconds")
        
        logger.info("\nDiscovered Pages:")
        for page in result['pages']:
            logger.info(f"\nURL: {page['url']}")
            if 'analysis' in page:
                analysis = page['analysis']
                logger.info("=== Page Analysis ===")
                if analysis.get('company'):
                    logger.info(f"Company Name: {analysis['company'].get('name')}")
                    logger.info(f"Description: {analysis['company'].get('description')}")
                
                if analysis.get('products_services'):
                    logger.info("\nProducts/Services:")
                    for product in analysis['products_services']:
                        logger.info(f"- {product.get('name')}: {product.get('description')}")
                
                if analysis.get('pricing_plans'):
                    logger.info("\nPricing Plans:")
                    for plan in analysis['pricing_plans']:
                        logger.info(f"- {plan.get('name')}: {plan.get('price')} {plan.get('period')}")
            
            if 'error' in page:
                logger.error(f"Error in page: {page['error']}")
            
            logger.info("-" * 80)
        
        return result
        
    except Exception as e:
        logger.error(f"Error during crawl: {str(e)}", exc_info=True)
        return None

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Web Crawler CLI')
    parser.add_argument('url', help='URL to crawl')
    parser.add_argument('--max-pages', type=int, default=10,
                      help='Maximum number of pages to crawl (default: 10)')
    parser.add_argument('--max-depth', type=int, default=2,
                      help='Maximum depth to crawl (default: 2)')
    
    args = parser.parse_args()
    
    # Run the crawler
    asyncio.run(run_crawler(args.url, args.max_pages, args.max_depth))

if __name__ == "__main__":
    main() 