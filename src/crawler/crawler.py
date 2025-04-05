from playwright.async_api import async_playwright, Error as PlaywrightError, Page
from langchain_deepseek.chat_models import ChatDeepSeek
from bs4 import BeautifulSoup
import json
import asyncio
import time
import os
from datetime import datetime
from typing import Dict, List, Any, Set, Tuple, Optional
from urllib.parse import urljoin, urlparse
from dotenv import load_dotenv
import logging
import re
from .state import crawl_progress # Assuming you have this state management module
from asyncio import Queue, Semaphore, Task, gather

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()

# Create results directory if it doesn't exist
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# --- Configuration ---
# Set to True to potentially speed up crawls by blocking non-essential resources.
# May break sites that rely heavily on JS for rendering text content.
BLOCK_RESOURCES = True
# Selectors to try for extracting main content, in order of preference.
# Add more selectors relevant to the sites you typically crawl.
TARGET_CONTENT_SELECTORS = [
    "main",
    "article",
    ".content",
    "#content",
    ".main-content",
    "#main-content",
    "body" # Fallback
]
# --- End Configuration ---

def extract_json_from_markdown(text: str) -> Dict[str, Any]:
    """Extract JSON from markdown code blocks or plain text."""
    json_match = re.search(r"```(?:json)?\n(.*?)\n```", text, re.DOTALL)
    if json_match:
        json_str = json_match.group(1).strip()
    else:
        # Remove potential markdown backticks if no code block found
        json_str = text.strip().lstrip('```json').lstrip('```').rstrip('```').strip()

    try:
        result = json.loads(json_str)
        # Basic validation: ensure it's a dictionary
        if not isinstance(result, dict):
             raise json.JSONDecodeError("Expected a JSON object (dict)", json_str, 0)
        return result
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse JSON, attempting cleanup: {e}\nRaw text: {text[:200]}...")
        # Try very basic cleanup (might need more robust handling)
        cleaned_str = re.sub(r'[^\x00-\x7F]+', '', json_str) # Remove non-ASCII chars
        cleaned_str = cleaned_str.replace('\\n', ' ').replace('\n', ' ') # Replace newlines
        try:
            # Ensure brackets/braces match - simplistic check
            if cleaned_str.count('{') == cleaned_str.count('}') and cleaned_str.count('[') == cleaned_str.count(']'):
                 result = json.loads(cleaned_str)
                 if not isinstance(result, dict):
                      raise json.JSONDecodeError("Expected a JSON object (dict)", cleaned_str, 0)
                 logger.info("Successfully parsed JSON after cleanup.")
                 return result
            else:
                 raise json.JSONDecodeError("Mismatched brackets/braces after cleanup", cleaned_str, 0)
        except json.JSONDecodeError as final_e:
            logger.error(f"Final JSON parsing failed: {final_e}\nCleaned text attempt: {cleaned_str[:200]}...")
            return {
                "error": f"Failed to parse JSON: {final_e}",
                "raw_response": text # Include raw response for debugging
            }

async def block_requests(route, request):
    """Block specified resource types."""
    if request.resource_type in ["image", "stylesheet", "font", "media"]:
        try:
            await route.abort()
        except PlaywrightError:
             # Can happen if request finishes before abort, ignore
             pass
    else:
        try:
            await route.continue_()
        except PlaywrightError:
             # Can happen if request finishes before continue, ignore
             pass

class WebCrawler:
    def __init__(self, task_id: str = None):
        self.llm = ChatDeepSeek(
            temperature=0.1, # Slightly more creative for summaries but still factual
            model="deepseek-chat", # Use your preferred model
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            api_base=os.getenv("DEEPSEEK_API_BASE"),
            # Consider adding timeout configuration for the LLM client if available
        )

        self.playwright = None
        self.browser = None
        self.visited_urls: Set[str] = set()
        self.start_time: Optional[float] = None
        self.task_id: str | None = task_id
        self.results: List[Dict[str, Any]] = []
        self.base_domain: str = ""
        self.queue: Queue[Tuple[str, int]] = Queue()
        self.semaphore: Semaphore | None = None
        self.max_concurrent_pages: int = 5 # Default concurrency limit

        if task_id:
            crawl_progress[task_id] = {
                "status": "initializing", "phase": "setup", "pages_crawled": 0,
                "urls_in_queue": 0, "current_url": None,
                "start_time": datetime.now().isoformat(),
                "last_update": datetime.now().isoformat(), "errors": []
            }

    def _update_progress(self, updates: Dict[str, Any]):
        if self.task_id:
            crawl_progress[self.task_id].update({
                **updates, "pages_crawled": len(self.visited_urls),
                "urls_in_queue": self.queue.qsize(),
                "last_update": datetime.now().isoformat()
            })

    async def setup(self):
        if self.playwright and self.browser: return
        self._update_progress({"status": "setting up browser"})
        try:
            self.playwright = await async_playwright().start()
            # Consider adding options like proxy settings if needed
            self.browser = await self.playwright.chromium.launch(headless=True)
            self.semaphore = Semaphore(self.max_concurrent_pages)
            self._update_progress({"status": "ready"})
            logger.info("Playwright setup complete.")
        except Exception as e:
            logger.error(f"Error during Playwright setup: {str(e)}")
            self._update_progress({"status": "failed", "error": f"Setup failed: {str(e)}"})
            raise

    async def cleanup(self):
        self._update_progress({"status": "cleaning up"})
        try:
            if self.browser: await self.browser.close(); self.browser = None
            if self.playwright: await self.playwright.stop(); self.playwright = None
            logger.info("Playwright cleanup complete.")
            self._update_progress({"status": "finished", "phase": "cleanup"})
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")

    async def _extract_focused_content(self, page: Page) -> str:
        """Try to extract text from main content areas first."""
        text_content = ""
        for selector in TARGET_CONTENT_SELECTORS:
            try:
                # Use inner_text which is often better for user-visible text
                locator = page.locator(selector)
                count = await locator.count()
                if count > 0:
                     # If multiple elements match (e.g., multiple <article>), concatenate their text
                     all_texts = await locator.all_inner_texts()
                     text_content = "\n\n".join(all_texts)
                     if text_content.strip():
                         logger.debug(f"Extracted content using selector: '{selector}'")
                         return text_content.strip()
            except PlaywrightError as e:
                logger.warning(f"Playwright error finding selector '{selector}': {e}")
            except Exception as e:
                 logger.warning(f"Error extracting text with selector '{selector}': {e}")

        # Fallback if no specific selectors worked (should always hit 'body')
        if not text_content.strip():
             logger.warning("No targeted content found, falling back to full body text (might be less accurate).")
             try:
                # Get text from body, excluding common noise tags via JS evaluation
                text_content = await page.evaluate("""() => {
                    const body = document.body.cloneNode(true);
                    body.querySelectorAll('script, style, nav, footer, header, noscript, svg, button, input, select, textarea, aside').forEach(el => el.remove());
                    return body.innerText;
                 }""")
             except Exception as e:
                  logger.error(f"Failed to extract fallback body text: {e}")
                  return "" # Return empty string if even fallback fails

        return text_content.strip()


    async def _process_page(self, url: str, depth: int) -> Optional[Dict[str, Any]]:
        if url in self.visited_urls: return None

        async with self.semaphore:
            page_start_time = time.time()
            context = None
            page = None
            try:
                self.visited_urls.add(url)
                self._update_progress({"current_url": url, "phase": f"processing depth {depth}"})
                logger.info(f"Processing [Depth {depth}]: {url}")

                # Use a single context with appropriate options
                # Consider settings like user agent, viewport, geolocation if needed
                context = await self.browser.new_context(
                    user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
                )
                page = await context.new_page()

                # Optional: Block resources for potentially faster loads
                if BLOCK_RESOURCES:
                    await page.route("**/*", block_requests)

                # Navigate (consider 'domcontentloaded' for speed vs 'load' for completeness)
                logger.debug(f"Navigating to {url}")
                try:
                    response = await page.goto(url, wait_until="load", timeout=60000) # Increased timeout slightly
                except PlaywrightError as e:
                     # Handle navigation errors more gracefully
                     raise Exception(f"Navigation error for {url}: {e}")

                if not response or not response.ok:
                    raise Exception(f"Failed to load {url}: Status {response.status if response else 'N/A'}")

                current_url = page.url
                if current_url != url:
                    logger.info(f"Redirected from {url} to {current_url}")
                    if current_url in self.visited_urls:
                        logger.info(f"Skipping already visited redirected URL: {current_url}")
                        return None
                    self.visited_urls.add(current_url)

                # Extract Focused Content (more efficient)
                logger.debug(f"Extracting focused content from {current_url}")
                text_content = await self._extract_focused_content(page)

                if not text_content:
                     logger.warning(f"No text content extracted from {current_url}. Skipping analysis.")
                     analysis_result = {"error": "No text content found on page"}
                     links = [] # Avoid link extraction if page content failed
                else:
                    # Analyze Content (using focused text)
                    logger.debug(f"Analyzing content of {current_url} (length: {len(text_content)})")
                    analysis_result = await self.analyze_page_content_for_sales(text_content, current_url)

                    # Extract Links (only if content extraction was successful)
                    logger.debug(f"Extracting links from {current_url}")
                    links = await self._extract_page_links(page, current_url)


                page_result = {
                    "url": current_url,
                    "depth": depth,
                    "analysis": analysis_result, # Contains sales-focused info or error
                    "links_found": links,
                    "crawl_time": time.time() - page_start_time,
                    "status": "success" if "error" not in analysis_result else "success_with_analysis_error"
                }
                # Note: We consider the page crawl successful even if LLM analysis fails,
                # but mark the analysis part separately.
                logger.info(f"Successfully processed (Analysis Status: {'OK' if 'error' not in analysis_result else 'Failed'}) [Depth {depth}]: {current_url}")
                return page_result

            except Exception as e:
                error_msg = f"Error processing page {url} (Depth {depth}): {str(e)}"
                logger.error(error_msg)
                self._update_progress({"errors": crawl_progress.get(self.task_id, {}).get("errors", []) + [error_msg]})
                return {
                    "url": url, "depth": depth, "status": "error",
                    "error": error_msg, "crawl_time": time.time() - page_start_time,
                    "analysis": None, "links_found": [] # Ensure keys exist even on error
                }
            finally:
                # Ensure cleanup happens even if errors occur mid-process
                if page:
                    try: await page.close()
                    except Exception as page_close_e: logger.warning(f"Error closing page for {url}: {page_close_e}")
                if context:
                    try: await context.close()
                    except Exception as context_close_e: logger.warning(f"Error closing context for {url}: {context_close_e}")


    async def _extract_page_links(self, page, current_page_url: str) -> List[str]:
        """Extract and filter links from the current page."""
        try:
            # Use Playwright's evaluation for efficiency
            raw_links = await page.eval_on_selector_all(
                'a[href]',
                'elements => elements.map(el => el.href)'
            )

            filtered_links: Set[str] = set()
            current_domain = urlparse(current_page_url).netloc

            for link in raw_links:
                try:
                    if not link or link.startswith(('#', 'mailto:', 'tel:', 'javascript:')): continue

                    absolute_link = urljoin(current_page_url, link)
                    parsed_link = urlparse(absolute_link)

                    if parsed_link.scheme not in ['http', 'https']: continue

                    link_domain = parsed_link.netloc
                    # Strict domain/subdomain matching based on initial base_domain
                    if not (link_domain == self.base_domain or link_domain.endswith('.' + self.base_domain)):
                        continue

                    # Ignore links pointing to common file types we usually don't want to crawl
                    if any(absolute_link.lower().endswith(ext) for ext in ['.pdf', '.jpg', '.png', '.zip', '.docx', '.xlsx', '.gif', '.svg', '.jpeg']):
                         continue

                    normalized_link = parsed_link._replace(fragment="", query="").geturl() # Remove query params too for stricter uniqueness

                    # Ensure trailing slash consistency (optional, but can reduce duplicates)
                    if normalized_link.endswith('/'):
                         normalized_link = normalized_link[:-1]

                    if normalized_link not in self.visited_urls:
                        filtered_links.add(normalized_link)

                except Exception as link_e:
                    logger.warning(f"Error processing link '{link}' on page {current_page_url}: {link_e}")
                    continue

            return list(filtered_links)

        except Exception as e:
            logger.error(f"Error extracting links from {current_page_url}: {str(e)}")
            return []

    async def analyze_page_content_for_sales(self, content: str, url: str) -> Dict[str, Any]:
        """Analyze page content using LLM, focusing on sales-relevant information."""
        # Limit content length to avoid excessive LLM costs/time, focus on relevance
        max_content_length = 8000 # Increased slightly as focused content might be longer
        content_to_analyze = content[:max_content_length]

        prompt = f"""
        Analyze the following text content from the webpage URL '{url}' for information relevant to a potential customer or sales agent. Extract the information in JSON format.

        Prioritize information about products, services, pricing, features, benefits, target audience, duration, requirements, company strengths (USPs), and contact details.

        JSON Schema:
        {{
          "page_topic": "Infer the main topic (e.g., 'Product/Service Page', 'About Us', 'Contact Us', 'Blog Post', 'General Info', 'Pricing Page', 'Testimonials')",
          "product_name": "Name of the specific product/service discussed, if applicable (string or null)",
          "company_name_mentioned": ["List of company names explicitly mentioned on this page"],
          "description_summary": "A brief summary of the page's main content or product description (string or null)",
          "pricing": {{
             "details": "Describe pricing if mentioned (e.g., '$X one-time', '$Y/month', 'Contact for quote') (string or null)",
             "currency": "Currency code (e.g., 'USD', 'EUR', 'ZAR') if price is specific (string or null)"
          }},
          "duration": "Duration if mentioned (e.g., '6 months', '4 weeks', 'Self-paced') (string or null)",
          "target_audience": "Describe the intended audience if mentioned (e.g., 'Beginners', 'Experienced developers', 'Office professionals') (string or null)",
          "prerequisites": "List any prerequisites or requirements mentioned (list of strings or null)",
          "key_features_or_benefits": ["List key features, benefits, or selling points mentioned (list of strings)"],
          "learning_outcomes": ["List expected learning outcomes or skills gained, if applicable (list of strings)"],
          "enrollment_info": "Describe how to enroll/apply/purchase, or next steps if mentioned (string or null)",
          "company_usp_mentioned": ["List any unique selling propositions or company strengths highlighted on this page (list of strings)"],
          "contact_email": "Primary contact email found on this page (string or null)",
          "contact_phone": "Primary contact phone number found on this page (string or null)",
          "extracted_from_url": "{url}"
        }}

        Instructions:
        - If specific information isn't found in the text, use null for strings/objects or empty lists ([]) for arrays.
        - Focus ONLY on the provided text content. Do not infer information not present.
        - Provide ONLY the JSON object, without any introductory text or markdown formatting.

        Text Content to Analyze (up to {max_content_length} chars):
        {content_to_analyze}
        """

        try:
            # Add timeout to LLM call if the library supports it
            response = await self.llm.ainvoke(prompt)
            if not response or not response.content:
                logger.warning(f"LLM returned no content for {url}")
                return {"error": "No response content from LLM", "extracted_from_url": url}

            # Use the robust JSON extraction function
            result = extract_json_from_markdown(response.content)

            # Add the source URL back if extraction was successful but didn't include it (should be in schema now)
            if "error" not in result and "extracted_from_url" not in result:
                 result["extracted_from_url"] = url

            if "error" in result:
                 logger.error(f"LLM analysis failed for {url}. Error: {result.get('error')}. Raw response: {result.get('raw_response', '')[:200]}...")
                 # Keep the error structure but ensure source URL is present
                 result["extracted_from_url"] = url
                 return result

            return result

        except asyncio.TimeoutError:
             logger.error(f"LLM analysis timed out for {url}")
             return {"error": "LLM analysis timed out", "extracted_from_url": url}
        except Exception as e:
            error_msg = f"Error during LLM analysis for {url}: {str(e)}"
            logger.exception(error_msg) # Log full traceback for unexpected errors
            self._update_progress({"errors": crawl_progress.get(self.task_id, {}).get("errors", []) + [error_msg]})
            return {"error": error_msg, "extracted_from_url": url}

    async def crawl(self, start_url: str, max_pages: int = 10, max_depth: int = 3, concurrency: int = 5) -> Dict[str, Any]:
        self.start_time = time.time()
        self.results = []
        self.visited_urls = set()
        self.max_concurrent_pages = concurrency
        try:
            parsed_start_url = urlparse(start_url)
            self.base_domain = parsed_start_url.netloc
            if not self.base_domain:
                 raise ValueError("Invalid start URL, could not determine domain.")
        except ValueError as e:
             logger.error(f"Invalid start URL provided: {start_url} - {e}")
             return {"error": f"Invalid start URL: {start_url}", "pages": []}

        self._update_progress({
            "status": "starting", "phase": "initializing crawl", "start_url": start_url,
            "max_pages": max_pages, "max_depth": max_depth, "concurrency": concurrency
        })

        await self.setup() # Ensure browser is ready

        await self.queue.put((start_url, 0))
        workers: List[Task] = []

        try:
            active_processing = True
            while active_processing:
                # Check stop conditions
                if len(self.visited_urls) >= max_pages:
                    logger.info(f"Reached max pages limit ({max_pages}). Stopping crawl.")
                    break # Stop adding new tasks

                # Check if we should stop adding tasks but let existing ones finish
                if self.queue.empty() and not workers:
                     logger.info("Queue is empty and all workers finished.")
                     active_processing = False # Exit loop after this iteration
                     continue # Let loop finish naturally


                # Launch new workers if capacity allows
                can_start_more = len(workers) < self.max_concurrent_pages
                pages_left_to_visit = max_pages - len(self.visited_urls)

                while not self.queue.empty() and can_start_more and pages_left_to_visit > 0:
                    try:
                        url, depth = self.queue.get_nowait() # Use non-blocking get
                        if url in self.visited_urls:
                            self.queue.task_done()
                            continue

                        if depth > max_depth:
                            logger.debug(f"Skipping {url} - Exceeded max depth ({depth}/{max_depth})")
                            self.queue.task_done()
                            continue

                        # Check visited again right before creating task (reduce race condition)
                        if url in self.visited_urls:
                             self.queue.task_done()
                             continue

                        logger.debug(f"Creating worker for: {url} (Depth: {depth})")
                        worker = asyncio.create_task(self._process_page(url, depth))
                        workers.append(worker)
                        self.queue.task_done()
                        pages_left_to_visit -= 1 # Decrement pages we intend to visit
                        can_start_more = len(workers) < self.max_concurrent_pages # Re-evaluate capacity

                    except asyncio.QueueEmpty:
                        break # No items currently in queue
                    except Exception as e:
                        logger.error(f"Error managing queue or starting worker: {e}")
                        # Ensure task_done is called if item was retrieved but worker failed
                        if 'url' in locals(): self.queue.task_done()
                        break # Avoid potential infinite loop on errors


                # Process finished workers
                if not workers:
                     # If no workers are running, and queue is empty, we should exit (handled by active_processing)
                     # If queue is not empty, loop will try to add workers again
                     await asyncio.sleep(0.1) # Prevent busy-waiting
                     continue

                # Wait for at least one worker to complete, with a timeout
                done, pending = await asyncio.wait(workers, return_when=asyncio.FIRST_COMPLETED, timeout=1.0)

                for task in done:
                    workers.remove(task)
                    try:
                         page_result = task.result()
                         if page_result:
                            self.results.append(page_result) # Store success or error result

                            # If successful and depth allows, add new links
                            if page_result.get("status") != "error" and page_result["depth"] < max_depth:
                                links_to_add = page_result.get("links_found", [])
                                queued_count = 0
                                for link in links_to_add:
                                    # Check max pages again before adding to queue
                                    # Estimate: visited + in queue + active workers
                                    estimated_total = len(self.visited_urls) + self.queue.qsize() + len(workers)
                                    if estimated_total < max_pages:
                                        # Check visited_urls *again* right before putting
                                        if link not in self.visited_urls:
                                            # Avoid adding duplicates to the queue itself if possible
                                            # This check is tricky with async queues, rely on visited_urls primarily
                                            await self.queue.put((link, page_result["depth"] + 1))
                                            queued_count += 1
                                    else:
                                        logger.debug(f"Max pages near limit ({max_pages}), not queueing more links from {page_result.get('url')}")
                                        break
                                if queued_count > 0:
                                     logger.debug(f"Added {queued_count} new links to queue from {page_result.get('url')}")

                    except asyncio.CancelledError:
                         logger.warning(f"Worker task for was cancelled.")
                    except Exception as task_exc:
                         logger.error(f"Worker task failed with exception: {task_exc}")
                         # Potentially store this failure if needed, but page_result should capture errors too
                # Small sleep if loop is very active
                await asyncio.sleep(0.05)


        except asyncio.CancelledError:
            logger.warning("Crawl task was cancelled.")
            self._update_progress({"status": "cancelled"})
        except Exception as e:
            error_msg = f"Unhandled exception during crawl loop: {str(e)}"
            logger.exception(error_msg)
            self._update_progress({"status": "failed", "error": error_msg})
        finally:
            # Cancel any remaining workers if crawl ended prematurely (e.g., max_pages)
            if workers:
                 logger.info(f"Cancelling {len(workers)} remaining worker tasks.")
                 for task in workers: task.cancel()
                 await gather(*workers, return_exceptions=True) # Wait for cancellations


            total_crawl_time = time.time() - self.start_time

            # --- Generate Final Output ---
            successful_pages = [r for r in self.results if r.get("status") != "error"]
            failed_pages_count = sum(1 for r in self.results if r.get("status") == "error")

            # Simple summary for RAG context
            crawl_summary = self._create_crawl_summary(successful_pages)

            final_output = {
                "crawl_metadata": {
                    "start_url": start_url,
                    "base_domain": self.base_domain,
                    "results_file": None, # Placeholder, will be updated below
                    "crawl_start_time": datetime.fromtimestamp(self.start_time).isoformat() if self.start_time else None,
                    "crawl_end_time": datetime.now().isoformat(),
                    "total_crawl_time_seconds": total_crawl_time,
                    "max_pages_limit": max_pages,
                    "max_depth_limit": max_depth,
                    "concurrency_limit": self.max_concurrent_pages,
                    "resource_blocking_enabled": BLOCK_RESOURCES,
                    "total_pages_processed": len(self.results),
                    "successful_pages_analyzed": len(successful_pages),
                    "failed_or_error_pages": failed_pages_count,
                },
                "crawl_summary": crawl_summary, # Basic aggregated info
                "pages": self.results # Detailed analysis per page (core data for RAG)
            }

            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"crawl_results_{self.base_domain}_{timestamp}.json"
            filepath = os.path.join(RESULTS_DIR, filename)

            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(final_output, f, indent=2, ensure_ascii=False)
                logger.info(f"Results saved to {filepath}")
                # Update the placeholder in the final output before returning
                final_output["crawl_metadata"]["results_file"] = filepath
                self._update_progress({
                    "status": "completed",
                    "phase": "saving results",
                    "results_file": filepath
                })
            except Exception as e:
                 logger.error(f"Failed to save results to {filepath}: {str(e)}")
                 self._update_progress({"status": "completed_with_save_error", "error": f"Failed to save results: {str(e)}"})

            await self.cleanup()
            return final_output


    def _create_crawl_summary(self, successful_pages: List[Dict[str, Any]]) -> Dict[str, Any]:
         """Creates a high-level summary from successful page analyses."""
         summary = {
             "identified_product_names": set(),
             "identified_company_names": set(),
             "contact_emails": set(),
             "contact_phones": set(),
             "page_topics_summary": {} # Count occurrences of page topics
         }
         if not successful_pages:
             return {k: list(v) if isinstance(v, set) else v for k, v in summary.items()} # Return empty lists/dicts

         for page_data in successful_pages:
             analysis = page_data.get("analysis")
             if not analysis or isinstance(analysis, dict) and analysis.get("error"):
                 continue # Skip pages with analysis errors

             # Ensure analysis is a dict before proceeding
             if not isinstance(analysis, dict):
                  logger.warning(f"Analysis for {page_data.get('url')} is not a dictionary: {type(analysis)}")
                  continue


             if analysis.get("product_name"):
                 summary["identified_product_names"].add(analysis["product_name"])

             if analysis.get("company_name_mentioned"):
                 for name in analysis["company_name_mentioned"]:
                     if name: summary["identified_company_names"].add(name.strip())

             if analysis.get("contact_email"):
                 summary["contact_emails"].add(analysis["contact_email"])
             if analysis.get("contact_phone"):
                 summary["contact_phones"].add(analysis["contact_phone"])

             page_topic = analysis.get("page_topic", "Unknown")
             summary["page_topics_summary"][page_topic] = summary["page_topics_summary"].get(page_topic, 0) + 1

         # Convert sets to lists for JSON serialization
         summary["identified_product_names"] = sorted(list(summary["identified_product_names"]))
         summary["identified_company_names"] = sorted(list(summary["identified_company_names"]))
         summary["contact_emails"] = sorted(list(summary["contact_emails"]))
         summary["contact_phones"] = sorted(list(summary["contact_phones"]))

         return summary


# --- Main Execution ---
async def crawl_website(url: str, max_pages: int = 10, max_depth: int = 3, concurrency: int = 5, task_id: str = None) -> Dict[str, Any]:
    """Run crawler with specified parameters."""
    crawler = WebCrawler(task_id=task_id)
    return await crawler.crawl(url, max_pages=max_pages, max_depth=max_depth, concurrency=concurrency)

if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(description="Async Web Crawler for Sales Agent RAG")
    parser.add_argument("url", help="Starting URL to crawl")
    parser.add_argument("-p", "--max_pages", type=int, default=25, help="Maximum number of pages to crawl (default: 25)")
    parser.add_argument("-d", "--max_depth", type=int, default=3, help="Maximum crawl depth (default: 3)")
    parser.add_argument("-c", "--concurrency", type=int, default=5, help="Number of concurrent pages to process (default: 5)")
    parser.add_argument("--no-block", action="store_false", dest="block_resources", help="Disable resource blocking (images, css, fonts)")
    parser.set_defaults(block_resources=BLOCK_RESOURCES) # Use configured default

    if len(sys.argv) == 1:
         # Simple example if no args provided
         print("Usage: python -m your_module.crawler <url> [options]")
         print("Running with default example: python -m your_module.crawler [https://www.zaio.io/](https://www.zaio.io/) -p 15 -d 2 -c 5")
         # --- Example Run ---
         example_url = "[https://www.zaio.io/](https://www.zaio.io/)"
         example_max_pages = 15
         example_max_depth = 2
         example_concurrency = 5
         BLOCK_RESOURCES = True # Use blocking for example
         # --- End Example Run ---
         results = asyncio.run(crawl_website(example_url, example_max_pages, example_max_depth, example_concurrency))

    else:
         args = parser.parse_args()
         # Set global based on args (if you keep the global config flag)
         BLOCK_RESOURCES = args.block_resources
         print(f"Starting crawl: URL={args.url}, Max Pages={args.max_pages}, Max Depth={args.max_depth}, Concurrency={args.concurrency}, Resource Blocking={BLOCK_RESOURCES}")
         results = asyncio.run(crawl_website(args.url, args.max_pages, args.max_depth, args.concurrency))


    # Optional: Print summary after crawl
    if results and "crawl_metadata" in results:
        print("\n--- Crawl Execution Summary ---")
        meta = results['crawl_metadata']
        print(f"Processed: {meta.get('total_pages_processed', 'N/A')} pages")
        print(f"Successful Analyses: {meta.get('successful_pages_analyzed', 'N/A')}")
        print(f"Failed/Error Pages: {meta.get('failed_or_error_pages', 'N/A')}")
        print(f"Total Time: {meta.get('total_crawl_time_seconds', 'N/A'):.2f} seconds")
        # Assuming results file path is now logged during execution or you add it back to metadata
        print("-----------------------------")
        summary = results.get('crawl_summary', {})
        print("\n--- Content Summary ---")
        print(f"Identified Products: {summary.get('identified_product_names', [])}")
        print(f"Identified Emails: {summary.get('contact_emails', [])}")
        print(f"Identified Phones: {summary.get('contact_phones', [])}")
        print("-----------------------")
    elif "error" in results:
         print(f"\nCrawl failed: {results['error']}")