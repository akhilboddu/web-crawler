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
from langchain_google_genai import ChatGoogleGenerativeAI # Add Gemini import

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
        json_str = text.strip().lstrip('`').rstrip('`').strip()

    try:
        result = json.loads(json_str)
        if not isinstance(result, dict):
             logger.warning(f"Parsed JSON is not a dict, type: {type(result)}. Returning error.")
             raise json.JSONDecodeError("Expected a JSON object (dict)", json_str, 0)
        return result
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse JSON from LLM response: {e}\nRaw text snippet: {text[:200]}...")
        # Attempt cleanup: find first '{' and last '}'
        start_brace = json_str.find('{')
        end_brace = json_str.rfind('}')
        if start_brace != -1 and end_brace != -1 and start_brace < end_brace:
            potential_json = json_str[start_brace:end_brace+1]
            try:
                result = json.loads(potential_json)
                if not isinstance(result, dict):
                      raise json.JSONDecodeError("Expected a JSON object (dict) after cleanup", potential_json, 0)
                logger.info("Successfully parsed JSON after basic cleanup.")
                return result
            except json.JSONDecodeError as final_e:
                logger.error(f"Final JSON parsing failed after cleanup: {final_e}\nCleaned text attempt: {potential_json[:200]}...")
                return { "error": f"Failed to parse JSON: {final_e}", "raw_response": text }
        else:
             return { "error": f"Failed to parse JSON, cleanup unsuccessful: {e}", "raw_response": text }

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
        # --- LLM Configuration ---
        self.llm = None
        llm_mode = os.getenv("LLM_MODE", "google").lower() # Default to google if not set
        logger.info(f"Configured LLM Mode: {llm_mode}")

        if llm_mode == "google":
            google_api_key = os.getenv("GOOGLE_API_KEY")
            if google_api_key:
                self.llm = ChatGoogleGenerativeAI(
                    model="gemini-1.5-flash-latest", # Or "gemini-pro"
                    google_api_key=google_api_key,
                    temperature=0.1,
                    convert_system_message_to_human=True
                )
                logger.info("Using Google Gemini LLM.")
            else:
                logger.warning("LLM_MODE set to 'google' but GOOGLE_API_KEY is not found in .env.")

        elif llm_mode == "deepseek":
            deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
            if deepseek_api_key:
                 self.llm = ChatDeepSeek(
                     temperature=0.1,
                     model="deepseek-chat",
                     api_key=deepseek_api_key,
                     api_base=os.getenv("DEEPSEEK_API_BASE"),
                 )
                 logger.info("Using DeepSeek LLM.")
            else:
                 logger.warning("LLM_MODE set to 'deepseek' but DEEPSEEK_API_KEY is not found in .env.")
        else:
             logger.warning(f"Invalid LLM_MODE specified: '{llm_mode}'. Please use 'google' or 'deepseek'. Falling back.")
             # Optional: Fallback logic or try default (e.g., google if key exists)
             google_api_key = os.getenv("GOOGLE_API_KEY")
             if google_api_key:
                 self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=google_api_key, temperature=0.1, convert_system_message_to_human=True)
                 logger.info("Falling back to Google Gemini LLM as default.")
             else: # Add a check for deepseek as a final fallback if google isn't available
                 deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
                 if deepseek_api_key:
                     self.llm = ChatDeepSeek(temperature=0.1, model="deepseek-chat", api_key=deepseek_api_key, api_base=os.getenv("DEEPSEEK_API_BASE"))
                     logger.info("Falling back to Deepseek LLM as default.")


        if not self.llm:
            raise ValueError(f"LLM could not be configured. Check LLM_MODE ('{llm_mode}') and ensure the corresponding API key (GOOGLE_API_KEY or DEEPSEEK_API_KEY) is set in your .env file.")

        self.playwright = None
        self.browser = None
        self.visited_urls: Set[str] = set()
        self.start_time: Optional[float] = None
        self.task_id: str | None = task_id
        self.results: List[Dict[str, Any]] = [] # Stores results from _process_page
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

                context = await self.browser.new_context(
                    user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
                )
                page = await context.new_page()

                if BLOCK_RESOURCES: await page.route("**/*", block_requests)

                logger.debug(f"Navigating to {url}")
                try:
                    response = await page.goto(url, wait_until="load", timeout=60000)
                except PlaywrightError as e: raise Exception(f"Navigation error for {url}: {e}")

                if not response or not response.ok: raise Exception(f"Failed to load {url}: Status {response.status if response else 'N/A'}")

                current_url = page.url
                normalized_current_url = current_url.rstrip('/')
                normalized_current_url = re.sub(r'^(https?://)www\.', r'\1', normalized_current_url)

                # Handle redirects (visited check done in main loop now)
                # Compare original fetched URL (passed as `url`) with the final `current_url`
                original_url_normalized = url.rstrip('/')
                original_url_normalized = re.sub(r'^(https?://)www\.', r'\1', original_url_normalized)
                if normalized_current_url != original_url_normalized:
                    logger.info(f"Redirected from {url} to {current_url}")
                    # Add the final redirected URL to visited set if the main loop didn't catch it
                    if normalized_current_url not in self.visited_urls:
                        self.visited_urls.add(normalized_current_url)
                        logger.debug(f"Added redirected URL {normalized_current_url} to visited set during processing.")

                # Extract Content
                logger.debug(f"Extracting focused content from {current_url}")
                text_content = await self._extract_focused_content(page)

                analysis_result = None
                crawlable_links = []
                targeted_social_links = {}

                # --- Link Extraction (Always run, even if no text content) ---
                logger.debug(f"Extracting links from {current_url}")
                crawlable_links, targeted_social_links = await self._extract_page_links(page, current_url)

                # --- LLM Analysis (Only if content exists) ---
                if not text_content:
                     logger.warning(f"No text content extracted from {current_url}. Skipping LLM analysis.")
                     analysis_result = {"error": "No text content found on page", "page_topic": "Unknown"} # Add topic default
                else:
                    logger.debug(f"Analyzing content of {current_url} (length: {len(text_content)}) with LLM")
                    # Pass the NEW prompt/schema to the LLM
                    analysis_result = await self.analyze_page_content_for_sales(text_content, current_url)


                # --- Prepare Result --- #
                page_result = {
                    "url": current_url,
                    "depth": depth,
                    "analysis": analysis_result, # Dict with NEW detailed schema data or error info
                    "links_found": crawlable_links, # Links to potentially queue (Original URLs)
                    "targeted_social_links": targeted_social_links, # Links found via direct extraction
                    "crawl_time": time.time() - page_start_time,
                    "status": "success" if isinstance(analysis_result, dict) and "error" not in analysis_result else ("success_no_content" if not text_content else "success_with_analysis_error")
                }
                # Main success/failure logging moved to crawl loop after task completion
                return page_result

            except Exception as e:
                error_msg = f"Error processing page {url} (Depth {depth}): {str(e)}"
                logger.exception(error_msg) # Log full traceback for easier debugging
                self._update_progress({"errors": crawl_progress.get(self.task_id, {}).get("errors", []) + [error_msg]})
                # Ensure structure matches success case for easier aggregation later
                return { "url": url, "depth": depth, "status": "error", "error": error_msg, "crawl_time": time.time() - page_start_time, "analysis": {"error": error_msg}, "links_found": [], "targeted_social_links": {} }
            finally:
                if page:
                    try:
                        await page.close()
                    except Exception:
                        pass
                if context:
                    try:
                        await context.close()
                    except Exception:
                        pass

    async def _extract_page_links(self, page: Page, current_page_url: str) -> Tuple[List[str], Dict[str, str]]:
        """Extract crawlable links and identify social media links (both via href check and targeted locators)."""
        social_domains = { # Used for broad domain checking in hrefs
            "linkedin.com": "linkedin", "twitter.com": "twitter", "x.com": "twitter",
            "facebook.com": "facebook", "instagram.com": "instagram", "youtube.com": "youtube"
        }
        # Precise locators for common social link patterns
        targeted_social_locators = {
            'linkedin': 'a[href*="linkedin.com/company/"], a[href*="linkedin.com/in/"]',
            'twitter': 'a[href*="twitter.com/"], a[href*="x.com/"]', # Check both
            'facebook': 'a[href*="facebook.com/"]',
            'instagram': 'a[href*="instagram.com/"]',
            'youtube': 'a[href*="youtube.com/channel/"], a[href*="youtube.com/user/"], a[href*="youtube.com/@"]' # Added @ handles
        }
        href_social_links = {} # Links found by checking domain in href
        targeted_social_links = {} # Links found via specific locators

        logger.debug(f"Extracting links from: {current_page_url} (Base Domain: {self.base_domain})")

        # --- Targeted Social Link Extraction (using locators) ---
        for site, locator in targeted_social_locators.items():
            try:
                # Use Playwright's recommended way to get first match or None
                 elements = await page.locator(locator).all()
                 if elements:
                     href = await elements[0].get_attribute('href')
                     if href:
                        abs_href = urljoin(current_page_url, href)
                        parsed_social = urlparse(abs_href)
                        if parsed_social.scheme in ['http', 'https'] and parsed_social.netloc:
                             if site not in targeted_social_links: # Store first found
                                  targeted_social_links[site] = abs_href
                                  logger.debug(f"Targeted Extraction found ({site}): {abs_href}")
            except PlaywrightError as pe:
                 # Ignore timeout errors as element might just not be present
                 if "Timeout" not in str(pe):
                      logger.warning(f"Playwright Error during targeted social link extraction for {site}: {pe}")
            except Exception as e:
                 logger.warning(f"Error during targeted social link extraction for {site}: {e}")

        # --- General Link Extraction and Filtering (href check) ---
        crawlable_links: Set[str] = set() # Store ORIGINAL absolute links found crawlable
        try:
            # Fetch all hrefs at once
            hrefs = await page.locator('a[href]').evaluate_all('elements => elements.map(el => el.href)')
            logger.debug(f"Found {len(hrefs)} raw hrefs via a[href]. Filtering...")

            if not self.base_domain:
                 # Fallback if base_domain somehow wasn't set
                 parsed_current = urlparse(current_page_url)
                 # Remove www. when setting base domain
                 self.base_domain = parsed_current.netloc.replace("www.", "")
                 logger.warning(f"Base domain was not set, derived '{self.base_domain}' from {current_page_url}")

            for i, link in enumerate(hrefs):
                # log_prefix = f"Link {i+1}/{len(hrefs)} ('{str(link)[:100]}...'):"

                try:
                    # Basic Exclusions
                    if not link or link.startswith( ('#', 'mailto:', 'tel:', 'javascript:') ):
                        # logger.debug(f"{log_prefix} Discarding (Basic Exclusion)")
                        continue

                    absolute_link = urljoin(current_page_url, link)
                    parsed_link = urlparse(absolute_link)

                    # Scheme Check
                    if parsed_link.scheme not in ['http', 'https']:
                        # logger.debug(f"{log_prefix} Discarding (Invalid Scheme)")
                        continue

                    link_domain = parsed_link.netloc

                    # Social Domain Check (href method)
                    is_social = False
                    for domain_pattern, site_key in social_domains.items():
                        if domain_pattern in link_domain:
                             if site_key not in href_social_links and site_key not in targeted_social_links:
                                 href_social_links[site_key] = absolute_link
                                 # logger.debug(f"{log_prefix} Identified social link via href check ({site_key}): {absolute_link}")
                             is_social = True
                             break
                    if is_social: continue

                    # Internal Link Filtering (Compare against normalized base_domain)
                    normalized_link_domain = link_domain.replace("www.", "")
                    if not (normalized_link_domain == self.base_domain or normalized_link_domain.endswith('.' + self.base_domain)):
                        # logger.debug(f"{log_prefix} Discarding (External Domain)")
                        continue

                    # File Extension Check
                    common_file_extensions = [
                        '.pdf', '.jpg', '.png', '.zip', '.docx', '.xlsx', '.gif', '.svg', '.jpeg',
                        '.mp4', '.avi', '.mov', '.mp3', '.wav', '.css', '.js', '.xml', '.rss',
                        '.dmg', '.exe', '.iso', '.ppt', '.pptx', '.tar', '.gz', '.webp', '.ico' # Added common web/icon types
                    ]
                    path_lower = parsed_link.path.lower()
                    if any(path_lower.endswith(ext) for ext in common_file_extensions):
                        # logger.debug(f"{log_prefix} Discarding (File Extension)")
                        continue

                    # Normalization for visited checks ONLY
                    normalized_link_for_check = parsed_link._replace(fragment="", query="").geturl()
                    if normalized_link_for_check.endswith('/'): normalized_link_for_check = normalized_link_for_check[:-1]
                    normalized_link_for_check = re.sub(r'^(https?://)www\.', r'\1', normalized_link_for_check)

                    # Visited / Self-Reference Check using normalized versions
                    normalized_current_url = current_page_url.rstrip('/')
                    normalized_current_url = re.sub(r'^(https?://)www\.', r'\1', normalized_current_url)

                    if normalized_link_for_check in self.visited_urls or normalized_link_for_check == normalized_current_url:
                        # logger.debug(f"{log_prefix} Discarding (Visited/Self)")
                        continue

                    # Add the ORIGINAL absolute link to the set for this page's results
                    crawlable_links.add(absolute_link)
                    # logger.debug(f"{log_prefix} Keeping for queue: {absolute_link}")

                except Exception as link_e:
                    # Log specific error for this link, but continue processing others
                    # logger.warning(f"{log_prefix} Error processing link: {link_e}")
                    pass # Reduce noise for minor link parse errors

        except Exception as e:
            # Log error for the general link extraction phase
            logger.error(f"Error during general link extraction from {current_page_url}: {str(e)}")

        # Combine social links (prioritize targeted ones found via specific locators)
        final_social_links = {**href_social_links, **targeted_social_links}

        logger.info(f"Link Extraction Summary for {current_page_url}: Found {len(crawlable_links)} crawlable, {len(final_social_links)} social links ({len(targeted_social_links)} targeted).")
        # Return the list of ORIGINAL absolute links to be potentially queued
        return list(crawlable_links), final_social_links


    async def analyze_page_content_for_sales(self, content: str, url: str) -> Dict[str, Any]:
        """Analyze page content using LLM based on the detailed 25-point schema."""
        max_content_length = 18000 # Increase context slightly for complex schema
        content_to_analyze = content[:max_content_length]
        if len(content) > max_content_length:
             logger.debug(f"Truncating content for LLM analysis from {len(content)} to {max_content_length} chars for {url}")

        # --- NEW 25-Point Schema and Prompt --- #
        prompt = f"""
        Analyze the text content from the webpage URL '{url}' and extract information relevant to a sales agent or potential customer. Structure the output strictly according to the JSON schema provided below. Find all relevant information present in the text.

        JSON Schema:
        {{ 
          "business_name": "Primary business name mentioned (string or null)",
          "tagline_slogan": "Company tagline or slogan if explicitly stated (string or null)",
          "short_description": "A brief (1-2 sentence) description of the company or its main offering (string or null)",
          "long_description_about_us": "More detailed company description, mission, vision, or \"About Us\" text found (string or null)",
          "unique_selling_proposition_usp": ["List key unique selling points or differentiators mentioned (list of strings)"],
          "products": [
            {{
              "name": "Product Name (string)",
              "description": "Product description (string or null)",
              "features": ["List of key features (list of strings)"],
              "image_urls_mentioned": ["List any specific image URLs mentioned in relation to this product (list of strings)"]
            }}
          ],
          "services": [
            {{
              "name": "Service Name (string)",
              "scope": "Description of the service scope (string or null)",
              "benefits": ["List of key benefits (list of strings)"],
              "delivery_method": "How the service is delivered (e.g., 'Online', 'On-site', 'Consultation') (string or null)"
            }}
          ],
          "pricing_tiers": [
            {{
              "tier_name": "Name of pricing tier/package (e.g., 'Basic', 'Pro', 'Enterprise') (string)",
              "price": "Price description (e.g., '$X/month', '$Y one-time', 'Contact for quote') (string or null)",
              "currency": "Currency code (e.g., 'USD', 'EUR') if specific price found (string or null)",
              "features_included": ["List of features specific to this tier (list of strings)"],
              "payment_terms": "Any mention of payment terms (e.g., 'Annual billing', 'Monthly') (string or null)"
            }}
          ],
          "free_trials_demos": "Describe any free trials or demos offered (string or null)",
          "customer_segments": ["List target customer groups/industries mentioned (list of strings)"],
          "use_cases_scenarios": ["Describe specific use cases or scenarios highlighted (list of strings)"],
          "pain_points_solved": ["List customer pain points the product/service addresses (list of strings)"],
          "customer_testimonials": [
            {{
              "quote": "The testimonial text (string)",
              "customer_name": "Name of the customer/company giving testimonial (string or null)",
              "source": "Where the testimonial was found (e.g., 'Website Text') (string)"
            }}
          ],
          "case_studies": [
            {{
              "title_or_summary": "Title or brief summary of the case study (string)",
              "customer_name": "Customer name if mentioned (string or null)",
              "problem": "Problem addressed (string or null)",
              "solution": "Solution provided (string or null)",
              "results_metrics": ["Key results or metrics mentioned (list of strings)"]
            }}
          ],
          "success_metrics_kpis_mentioned": ["General success metrics or KPIs highlighted by the company (e.g., '99% uptime', 'ROI in 6 months') (list of strings)"],
          "guarantees_return_policy": "Describe any guarantees or return policies mentioned (string or null)",
          "onboarding_process_description": "Describe the customer onboarding process if mentioned (string or null)",
          "implementation_timelines": "Mention any estimated implementation timelines (string or null)",
          "support_channels": ["List available customer support channels mentioned (e.g., 'Chat', 'Email', 'Phone', 'Knowledge Base') (list of strings)"],
          "sales_contact_info": {{
            "email": "Sales-specific contact email found in text (string or null)",
            "phone": "Sales-specific contact phone found in text (string or null)",
            "contact_form_mention": "Is a sales contact form mentioned? (boolean or null)"
          }},
          "faqs": [
            {{
              "question": "The question text (string)",
              "answer": "The answer text (string)"
            }}
          ],
          "social_media_links_on_page": {{ "linkedin": null, "twitter": null, "facebook": null, "instagram": null, "youtube": null, "other": [] }},
          "press_mentions_media_coverage": ["List any press mentions or media outlets cited (list of strings)"],
          "reviews_ratings_mentioned": ["Describe any mentions of external reviews/ratings (e.g., '5 stars on G2', 'Trustpilot rating') (list of strings)"],
          "integrations_api_availability": "Describe mentions of integrations or API availability (string or null)",
          "page_topic": "Infer the main topic of this specific page (e.g., 'Homepage', 'Pricing', 'Product Detail', 'About Us', 'Case Study', 'Blog Post') (string)",
          "extracted_from_url": "{url}"
        }}

        Instructions:
        - Respond ONLY with the JSON object matching the schema. Do NOT include any introductory text, closing remarks, or markdown formatting like ```json or ```.
        - If a field is not found in the text, use null for strings/objects or an empty list ([]) for arrays.
        - If a page lists multiple products, services, pricing tiers, testimonials, case studies, or FAQs, extract ALL of them into their respective lists within the JSON. Be thorough.
        - For `social_media_links_on_page`, only extract URLs found directly *in the text content provided*. Do not guess based on icons if the URL isn't explicitly written or linked in the text.
        - Ensure all extracted text is presented cleanly without excessive whitespace or artifacts.

        Text Content to Analyze (up to {max_content_length} chars):
        {content_to_analyze}
        """

        try:
            logger.debug(f"Invoking LLM for page analysis: {url}")
            # Note: Consider adding a timeout specific to the LLM call if the library supports it.
            response = await self.llm.ainvoke(prompt)
            if not response or not response.content:
                logger.warning(f"LLM returned no content for {url}")
                # Return structure consistent with error cases
                return {"error": "No response content from LLM", "extracted_from_url": url, "page_topic": "Unknown"}

            result = extract_json_from_markdown(response.content)

            # Ensure extracted_from_url is present, even if LLM misses it
            if "error" not in result and "extracted_from_url" not in result:
                 result["extracted_from_url"] = url
            # Ensure page_topic has a default if missing
            if "page_topic" not in result: result["page_topic"] = "Unknown"

            if "error" in result:
                 logger.error(f"LLM analysis failed for {url}. Error: {result.get('error')}. Raw response snippet: {result.get('raw_response', '')[:200]}...")
                 # Add defaults to error structure for consistency
                 result["extracted_from_url"] = url
                 if "page_topic" not in result: result["page_topic"] = "Unknown"
                 return result

            # Minimal validation (ensure required list fields are lists if present)
            for list_field in ["products", "services", "pricing_tiers", "customer_testimonials", "case_studies", "faqs", "unique_selling_proposition_usp", "customer_segments", "use_cases_scenarios", "pain_points_solved", "success_metrics_kpis_mentioned", "support_channels", "press_mentions_media_coverage", "reviews_ratings_mentioned"]:
                 if list_field in result and not isinstance(result[list_field], list):
                     logger.warning(f"Field '{list_field}' in LLM response for {url} is not a list, attempting to fix or clear.")
                     # Basic attempt to fix if it's a single dict/str when it should be a list
                     if result[list_field] is not None: result[list_field] = [result[list_field]]
                     else: result[list_field] = [] # Clear if None or not fixable

            return result

        except asyncio.TimeoutError:
             logger.error(f"LLM analysis timed out for {url}")
             return {"error": "LLM analysis timed out", "extracted_from_url": url, "page_topic": "Unknown"}
        except Exception as e:
            error_msg = f"Error during LLM analysis for {url}: {str(e)}"
            logger.exception(error_msg)
            self._update_progress({"errors": crawl_progress.get(self.task_id, {}).get("errors", []) + [error_msg]})
            return {"error": error_msg, "extracted_from_url": url, "page_topic": "Unknown"}


    async def _aggregate_business_profile(self) -> Dict[str, Any]:
        """Aggregates data from page analyses based on the detailed 25-point schema."""
        profile = {
            # Initialize all 25 fields plus helpers
            "business_name": None, "tagline_slogan": None, "short_description": None,
            "long_description_about_us": None, "unique_selling_proposition_usp": [],
            "products": [], "services": [], "pricing_tiers": [], "free_trials_demos": None,
            "customer_segments": [], "use_cases_scenarios": [], "pain_points_solved": [],
            "customer_testimonials": [], "case_studies": [], "success_metrics_kpis_mentioned": [],
            "guarantees_return_policy": None, "onboarding_process_description": None,
            "implementation_timelines": None, "support_channels": [],
            "sales_contact_info": { "email": None, "phone": None, "contact_form_mention": None },
            "faqs": [], "social_media_links": {}, "press_mentions_media_coverage": [],
            "reviews_ratings_mentioned": [], "integrations_api_availability": None,
            # --- Helper/Meta fields ---
            "source_urls": [], "aggregated_company_summary": None, "base_domain": self.base_domain
        }

        # Sets to track uniqueness of complex items (using representative keys)
        product_seen = set()        # Use lower case name
        service_seen = set()        # Use lower case name
        pricing_tier_seen = set()   # Use lower case name
        testimonial_seen = set()    # Use lower case quote start
        case_study_seen = set()     # Use lower case title start
        faq_seen = set()            # Use lower case question start
        social_links_combined = {}  # Stores final unique social links {site: url}
        all_page_summaries = []     # Collects descriptions for final summary generation

        successful_pages = [r for r in self.results if r.get("status", "").startswith("success") and isinstance(r.get("analysis"), dict) and not r.get("analysis", {}).get("error")]

        if not successful_pages:
            logger.warning("No successful pages with analysis found for aggregation.")
            return profile # Return initialized profile

        # --- Prioritize certain pages for single-value fields --- #
        page_priority = sorted(successful_pages, key=lambda p: (
            p.get('depth', 99),
            0 if p.get('analysis', {}).get('page_topic') == 'Homepage' else
            1 if 'About' in p.get('analysis', {}).get('page_topic', '') else
            2 if 'Contact' in p.get('analysis', {}).get('page_topic', '') else
            3
           )
        )
        logger.info(f"Starting aggregation from {len(successful_pages)} successful pages. Prioritizing single-value fields.")

        # --- Pass 1: Extract single-value fields from prioritized pages --- #
        processed_single_value_fields = set() # Track which fields are set
        single_value_fields = [
            "business_name", "tagline_slogan", "short_description", "long_description_about_us",
            "free_trials_demos", "guarantees_return_policy", "onboarding_process_description",
            "implementation_timelines", "integrations_api_availability"
        ]

        for page_data in page_priority:
            analysis = page_data["analysis"]
            page_topic = analysis.get("page_topic", "Unknown")
            page_url = page_data.get("url", "unknown")

            # Extract single-value string fields
            for field in single_value_fields:
                if field not in processed_single_value_fields and analysis.get(field):
                    profile[field] = analysis[field]
                    processed_single_value_fields.add(field)
                    logger.debug(f"Set '{field}' from {page_topic} page: {page_url}")

            # Sales Contact - take first non-null component found
            sales_info = analysis.get("sales_contact_info", {})
            if isinstance(sales_info, dict): # Ensure it's a dict
                if "sales_contact_info_email" not in processed_single_value_fields and sales_info.get("email"):
                    profile["sales_contact_info"]["email"] = sales_info["email"]
                    processed_single_value_fields.add("sales_contact_info_email")
                    logger.debug(f"Set 'sales_contact_info.email' from {page_topic} page: {page_url}")
                if "sales_contact_info_phone" not in processed_single_value_fields and sales_info.get("phone"):
                    profile["sales_contact_info"]["phone"] = sales_info["phone"]
                    processed_single_value_fields.add("sales_contact_info_phone")
                    logger.debug(f"Set 'sales_contact_info.phone' from {page_topic} page: {page_url}")
                if "sales_contact_info_form" not in processed_single_value_fields and sales_info.get("contact_form_mention") is not None:
                    profile["sales_contact_info"]["contact_form_mention"] = sales_info.get("contact_form_mention")
                    processed_single_value_fields.add("sales_contact_info_form")
                    logger.debug(f"Set 'sales_contact_info.contact_form_mention' from {page_topic} page: {page_url}")

            # Stop early if all single-value fields are found
            if len(processed_single_value_fields) >= len(single_value_fields) + 3: # +3 for sales info components
                 logger.info("Found primary values early in prioritized pages.")
                 break

        # Fallback for business name
        if not profile["business_name"] and self.base_domain:
            domain_parts = self.base_domain.split('.')
            name_candidate = domain_parts[-2].capitalize() if len(domain_parts) >= 2 else domain_parts[0].capitalize()
            profile["business_name"] = name_candidate
            logger.info(f"Company name not explicitly found, derived '{profile['business_name']}' from domain {self.base_domain}.")

        # --- Pass 2: Aggregate lists and combine data from all pages --- #
        logger.info("Aggregating list-based fields from all successful pages.")
        temp_usp = set()
        temp_segments = set()
        temp_use_cases = set()
        temp_pain_points = set()
        temp_metrics = set()
        temp_support = set()
        temp_press = set()
        temp_reviews = set()

        for page_data in successful_pages:
            analysis = page_data["analysis"]
            page_url = page_data.get("url", "unknown_url")
            if page_url not in profile["source_urls"]: profile["source_urls"].append(page_url) # Track unique URLs

            # Collect summaries for final LLM call (use short/long description)
            if analysis.get("short_description"): all_page_summaries.append(analysis["short_description"])
            if analysis.get("long_description_about_us"): all_page_summaries.append(analysis["long_description_about_us"])

            # --- Aggregate Simple Lists (using sets for uniqueness) --- #
            def update_set_from_list(target_set, source_list):
                if isinstance(source_list, list):
                    target_set.update(str(item) for item in source_list if item) # Add non-empty stringified items

            update_set_from_list(temp_usp, analysis.get("unique_selling_proposition_usp"))
            update_set_from_list(temp_segments, analysis.get("customer_segments"))
            update_set_from_list(temp_use_cases, analysis.get("use_cases_scenarios"))
            update_set_from_list(temp_pain_points, analysis.get("pain_points_solved"))
            update_set_from_list(temp_metrics, analysis.get("success_metrics_kpis_mentioned"))
            update_set_from_list(temp_support, analysis.get("support_channels"))
            update_set_from_list(temp_press, analysis.get("press_mentions_media_coverage"))
            update_set_from_list(temp_reviews, analysis.get("reviews_ratings_mentioned"))

            # --- Aggregate Complex Lists (Products, Services, etc.) --- #
            # Products
            for product in analysis.get("products", []):
                if isinstance(product, dict) and product.get("name"):
                     product_key = product["name"].lower().strip()
                     if product_key and product_key not in product_seen:
                         product['source_url'] = page_url # Add source URL
                         profile["products"].append(product)
                         product_seen.add(product_key)

            # Services
            for service in analysis.get("services", []):
                if isinstance(service, dict) and service.get("name"):
                     service_key = service["name"].lower().strip()
                     if service_key and service_key not in service_seen:
                         service['source_url'] = page_url
                         profile["services"].append(service)
                         service_seen.add(service_key)

            # Pricing Tiers
            for tier in analysis.get("pricing_tiers", []):
                if isinstance(tier, dict) and tier.get("tier_name"):
                     tier_key = tier["tier_name"].lower().strip()
                     if tier_key and tier_key not in pricing_tier_seen:
                         tier['source_url'] = page_url
                         profile["pricing_tiers"].append(tier)
                         pricing_tier_seen.add(tier_key)

            # Testimonials
            for testimonial in analysis.get("customer_testimonials", []):
                if isinstance(testimonial, dict) and testimonial.get("quote"):
                     quote_key = testimonial["quote"].strip()[:100].lower() # Key based on quote start
                     if quote_key and quote_key not in testimonial_seen:
                         testimonial['source_url'] = page_url
                         profile["customer_testimonials"].append(testimonial)
                         testimonial_seen.add(quote_key)

            # Case Studies
            for study in analysis.get("case_studies", []):
                if isinstance(study, dict) and study.get("title_or_summary"):
                    study_key = study["title_or_summary"].strip()[:100].lower()
                    if study_key and study_key not in case_study_seen:
                         study['source_url'] = page_url
                         profile["case_studies"].append(study)
                         case_study_seen.add(study_key)

            # FAQs
            for faq in analysis.get("faqs", []):
                 if isinstance(faq, dict) and faq.get("question"):
                    faq_key = faq["question"].strip()[:100].lower()
                    if faq_key and faq_key not in faq_seen:
                         faq['source_url'] = page_url
                         profile["faqs"].append(faq)
                         faq_seen.add(faq_key)

            # --- Aggregate Social Links (Merge LLM + Targeted) --- #
            llm_socials = analysis.get("social_media_links_on_page", {})
            targeted_socials = page_data.get("targeted_social_links", {})
            current_page_socials = {}
            if isinstance(llm_socials, dict): current_page_socials.update(llm_socials)
            if isinstance(targeted_socials, dict): current_page_socials.update(targeted_socials) # Targeted overwrites LLM

            for site, url_val in current_page_socials.items():
                site_key = site.lower().strip()
                if not site_key or site_key == 'other' or not isinstance(url_val, str): continue

                if url_val and site_key not in social_links_combined:
                     try:
                         parsed_s = urlparse(url_val)
                         if parsed_s.scheme in ['http', 'https'] and parsed_s.netloc:
                              social_links_combined[site_key] = url_val
                     except Exception: pass # Ignore invalid URLs silently

        # --- Final Cleanup & Assignment --- #
        profile["unique_selling_proposition_usp"] = sorted(list(temp_usp))
        profile["customer_segments"] = sorted(list(temp_segments))
        profile["use_cases_scenarios"] = sorted(list(temp_use_cases))
        profile["pain_points_solved"] = sorted(list(temp_pain_points))
        profile["success_metrics_kpis_mentioned"] = sorted(list(temp_metrics))
        profile["support_channels"] = sorted(list(temp_support))
        profile["press_mentions_media_coverage"] = sorted(list(temp_press))
        profile["reviews_ratings_mentioned"] = sorted(list(temp_reviews))
        profile["social_media_links"] = social_links_combined # Assign combined dict
        profile["source_urls"] = sorted(list(set(profile["source_urls"]))) # Unique contributing URLs

        # Sort complex lists by name/title/question for consistency
        profile["products"].sort(key=lambda x: x.get('name', '').lower())
        profile["services"].sort(key=lambda x: x.get('name', '').lower())
        profile["pricing_tiers"].sort(key=lambda x: x.get('tier_name', '').lower())
        profile["customer_testimonials"].sort(key=lambda x: x.get('quote', '').lower())
        profile["case_studies"].sort(key=lambda x: x.get('title_or_summary', '').lower())
        profile["faqs"].sort(key=lambda x: x.get('question', '').lower())

        # --- Generate Aggregated Summary --- #
        unique_page_summaries = list(dict.fromkeys(filter(None, all_page_summaries))) # Deduplicate
        if unique_page_summaries:
            logger.info(f"Generating aggregated company summary from {len(unique_page_summaries)} unique description snippets.")
            profile["aggregated_company_summary"] = await self._summarize_company(unique_page_summaries)
        else:
             logger.warning("No descriptions found to generate aggregated company summary.")
             profile["aggregated_company_summary"] = None

        logger.info(f"Aggregation complete. Profile generated for {profile.get('business_name')}")
        return profile


    async def _summarize_company(self, summaries: List[str]) -> Optional[str]:
        """Generates an overall company summary using the LLM based on collected page summaries."""
        if not summaries: return None
        # Use unique summaries passed from aggregation
        max_summary_input_length = 10000
        combined_text = "\n\n---\n\n".join(summaries) # Already unique
        truncated_text = combined_text[:max_summary_input_length]

        if len(combined_text) > max_summary_input_length:
             logger.debug(f"Truncating combined summaries from {len(combined_text)} to {max_summary_input_length} for final summary LLM call.")

        prompt = f"""Based on the following descriptions extracted from a company's website, generate a concise (1-3 sentences) overall summary of the company. Focus on what the company *is* or *does*.

        Extracted Descriptions:
        ---
        {truncated_text}
        ---

        Overall Company Summary (1-3 sentences):"""

        try:
            logger.debug("Invoking LLM for final company summary.")
            response = await self.llm.ainvoke(prompt)
            if response and response.content:
                summary = response.content.strip().replace('"', '').replace("Overall Company Summary:", "").strip()
                summary = re.sub(r'^```(?:json)?\s*', '', summary, flags=re.MULTILINE)
                summary = re.sub(r'\s*```$', '', summary, flags=re.MULTILINE)
                summary = summary.strip()

                logger.info(f"Generated company summary: {summary[:150]}...")
                return summary
            else:
                logger.warning("LLM returned no content for company summary generation.")
                return None
        except asyncio.TimeoutError:
            logger.error("LLM call timed out during company summary generation.")
            return None
        except Exception as e:
            logger.error(f"Error during LLM company summary generation: {e}")
            return None


    async def crawl(self, start_url: str, max_pages: int = 10, max_depth: int = 3, concurrency: int = 5) -> Dict[str, Any]:
        self.start_time = time.time()
        self.results = []
        self.visited_urls: Set[str] = set() # Stores normalized URLs
        self.max_concurrent_pages = concurrency
        try:
            # Normalize start URL before parsing
            normalized_start_url_for_queue = start_url.strip().rstrip('/')
            if not normalized_start_url_for_queue.startswith(('http://', 'https://')):
                 normalized_start_url_for_queue = 'http://' + normalized_start_url_for_queue # Assume http if missing

            parsed_start_url = urlparse(normalized_start_url_for_queue)
            # Store base domain without www. for internal comparisons
            self.base_domain = parsed_start_url.netloc.replace("www.", "")
            if not self.base_domain or '.' not in self.base_domain:
                 raise ValueError("Invalid start URL, could not determine domain.")
            logger.info(f"Crawl starting for base domain: {self.base_domain}")
        except ValueError as e:
             logger.error(f"Invalid start URL provided: {start_url} - {e}")
             return {"error": f"Invalid start URL: {start_url}", "pages": []}

        # --- Progress Update and Setup ---
        self._update_progress({
            "status": "starting", "phase": "initializing crawl", "start_url": start_url,
            "max_pages": max_pages, "max_depth": max_depth, "concurrency": concurrency
        })
        await self.setup()

        # --- Initialize Queue & Workers ---
        # Add the potentially non-normalized start URL to the queue
        await self.queue.put((normalized_start_url_for_queue, 0))

        workers: List[Task] = []

        try:
            active_processing = True
            while active_processing:
                # --- Stop Condition Checks --- #
                if len(self.visited_urls) >= max_pages:
                    logger.info(f"Reached max pages limit ({max_pages}). Stopping crawl task generation.")
                    break

                if self.queue.empty() and not workers:
                     logger.info("Queue is empty and all workers finished.")
                     active_processing = False
                     continue

                # --- Launch New Workers --- #
                can_start_more = len(workers) < self.max_concurrent_pages
                # Estimate pages remaining accurately
                pages_goal_remaining = max_pages - len(self.visited_urls)

                while not self.queue.empty() and can_start_more and pages_goal_remaining > 0:
                    try:
                        # Get the original URL (potentially with www.) from the queue
                        original_url_to_process, depth = self.queue.get_nowait()

                        # Normalize the URL *only* for the visited check
                        normalized_url_for_check = original_url_to_process.rstrip('/')
                        normalized_url_for_check = re.sub(r'^(https?://)www\.', r'\1', normalized_url_for_check)

                        # Check if the *normalized* version has been visited
                        if normalized_url_for_check in self.visited_urls:
                            self.queue.task_done()
                            continue # Already processed or currently being processed

                        if depth > max_depth:
                            logger.debug(f"Skipping {original_url_to_process} - Exceeded max depth ({depth}/{max_depth})")
                            self.queue.task_done()
                            continue

                        # Check visited again (redundant but safe)
                        if normalized_url_for_check in self.visited_urls:
                             self.queue.task_done()
                             continue

                        # Add normalized URL to visited *before* creating task
                        self.visited_urls.add(normalized_url_for_check)
                        pages_goal_remaining -= 1 # Decrement goal remaining

                        # Create worker, passing the ORIGINAL URL for fetching
                        logger.debug(f"Creating worker for: {original_url_to_process} (Depth: {depth}) - Added {normalized_url_for_check} to visited.")
                        worker = asyncio.create_task(self._process_page(original_url_to_process, depth))
                        workers.append(worker)
                        self.queue.task_done()
                        can_start_more = len(workers) < self.max_concurrent_pages

                    except asyncio.QueueEmpty:
                        break
                    except Exception as e:
                        logger.error(f"Error managing queue or starting worker: {e}")
                        # If we failed to start worker, potentially remove from visited?
                        if 'normalized_url_for_check' in locals() and normalized_url_for_check in self.visited_urls:
                            try: self.visited_urls.remove(normalized_url_for_check) ; pages_goal_remaining += 1
                            except KeyError: pass
                        if 'original_url_to_process' in locals(): self.queue.task_done()
                        break

                # --- Process Finished Workers --- #
                if not workers:
                     if not self.queue.empty(): await asyncio.sleep(0.1)
                     continue

                done, pending = await asyncio.wait(workers, return_when=asyncio.FIRST_COMPLETED, timeout=1.0)

                for task in done:
                    workers.remove(task)
                    try:
                         page_result = task.result()
                         if page_result:
                            # Log processed result (using the URL returned by _process_page, which might be redirected)
                            processed_url = page_result.get('url', 'UNKNOWN_URL')
                            page_depth = page_result.get('depth', -1)
                            page_status = page_result.get('status', 'UNKNOWN_STATUS')
                            analysis_status = "OK" if isinstance(page_result.get('analysis'), dict) and "error" not in page_result['analysis'] else "ERROR/MISSING"
                            links_found_count = len(page_result.get('links_found', []))
                            logger.info(f"Processed Result: URL={processed_url} | Depth={page_depth} | Status={page_status} | Analysis={analysis_status} | Links Found={links_found_count}")

                            self.results.append(page_result)
                            self._update_progress({}) # Update general progress state

                            # --- Add New Links to Queue --- #
                            if page_result.get("status", "").startswith("success") and page_result.get("depth", max_depth + 1) < max_depth:
                                # links_to_add contains ORIGINAL URLs (potentially with www.)
                                links_to_add = page_result.get("links_found", [])
                                links_actually_queued = 0
                                for original_link_to_queue in links_to_add:
                                    # Normalize link *only* for visited check
                                    normalized_link_for_check = original_link_to_queue.rstrip('/')
                                    normalized_link_for_check = re.sub(r'^(https?://)www\.', r'\1', normalized_link_for_check)

                                    # Estimate pages needed vs limit
                                    estimated_visited_count = len(self.visited_urls)
                                    if estimated_visited_count < max_pages:
                                        # Check *normalized* version against visited set
                                        if normalized_link_for_check not in self.visited_urls:
                                            # Check queue to avoid adding exact same original URL multiple times? (Less critical)
                                            # Check if already in queue? (Queue doesn't support easy check)

                                            # Queue the ORIGINAL link
                                            await self.queue.put((original_link_to_queue, page_result["depth"] + 1))
                                            links_actually_queued += 1
                                            # Don't add to visited here; wait until it's dequeued.
                                    else:
                                        logger.debug(f"Max pages limit ({max_pages}) reached/exceeded, not queueing more links from {processed_url}")
                                        break # Stop queueing from this page

                                if links_actually_queued > 0:
                                     logger.debug(f"Queued {links_actually_queued} new links from {processed_url}")

                    except asyncio.CancelledError:
                         logger.warning(f"Worker task was cancelled.")
                    except Exception as task_exc:
                         logger.error(f"Worker task failed with exception: {task_exc}")
                         # Log error in progress state
                         self._update_progress({"errors": crawl_progress.get(self.task_id, {}).get("errors", []) + [f"Worker task failed: {task_exc}"]})

                # Small sleep if loop is very active but nothing finished
                if not done and workers:
                    await asyncio.sleep(0.05)

        except asyncio.CancelledError:
            logger.warning("Crawl task was cancelled.")
            self._update_progress({"status": "cancelled"})
        except Exception as e:
            error_msg = f"Unhandled exception during crawl loop: {str(e)}"
            logger.exception(error_msg)
            self._update_progress({"status": "failed", "error": error_msg})
        finally:
            # Cancel any remaining workers if crawl ended prematurely
            if workers:
                 logger.info(f"Cancelling {len(workers)} remaining worker tasks.")
                 for task in workers: task.cancel()
                 await gather(*workers, return_exceptions=True)

            total_crawl_time = time.time() - self.start_time

            # --- Generate Final Output --- #
            successful_pages = [r for r in self.results if r.get("status", "").startswith("success") and isinstance(r.get("analysis"), dict) and not r.get("analysis", {}).get("error")]
            failed_pages_count = sum(1 for r in self.results if r.get("status") == "error")

            logger.info("Aggregating results into final business profile...")
            # *** Call the updated aggregation function ***
            aggregated_profile = await self._aggregate_business_profile()

            final_output = {
                "crawl_metadata": {
                    "start_url": start_url,
                    "base_domain": self.base_domain,
                    "results_file": None,
                    "crawl_start_time": datetime.fromtimestamp(self.start_time).isoformat() if self.start_time else None,
                    "crawl_end_time": datetime.now().isoformat(),
                    "total_crawl_time_seconds": total_crawl_time,
                    "max_pages_limit": max_pages,
                    "max_depth_limit": max_depth,
                    "concurrency_limit": self.max_concurrent_pages,
                    "resource_blocking_enabled": BLOCK_RESOURCES,
                    "total_pages_processed": len(self.results), # Pages task finished for
                    "total_urls_visited": len(self.visited_urls), # Unique URLs added to visited set
                    "successful_pages_analyzed": len(successful_pages),
                    "failed_or_error_pages": failed_pages_count,
                    "final_queue_size": self.queue.qsize()
                },
                "business_profile": aggregated_profile, # Use the new detailed profile
                "pages": self.results # Keep raw page data if needed
            }
            # --- Save Results --- #
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_base_domain = re.sub(r'[^\w\-.]', '_', self.base_domain or "unknown_domain")
            filename = f"crawl_results_{safe_base_domain}_{timestamp}.json"
            filepath = os.path.join(RESULTS_DIR, filename)

            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(final_output, f, indent=2, ensure_ascii=False)
                logger.info(f"Results saved to {filepath}")
                final_output["crawl_metadata"]["results_file"] = filepath
                self._update_progress({"status": "completed", "phase": "saving results", "results_file": filepath})
            except Exception as e:
                 logger.error(f"Failed to save results to {filepath}: {str(e)}")
                 self._update_progress({"status": "completed_with_save_error", "error": f"Failed to save results: {str(e)}"})

            await self.cleanup()
            return final_output

    # --- Old Crawl Summary (Can be removed or kept for basic overview) --- #
    def _create_crawl_summary(self, successful_pages: List[Dict[str, Any]]) -> Dict[str, Any]:
        # ... (This function uses the OLD schema, might be inaccurate now)
        # Consider removing or adapting if keeping
        logger.warning("_create_crawl_summary is based on the old schema and may be inaccurate or removed in the future.")
        # Return empty dict or adapt if needed
        return {}

# Helper function needed to run the crawler
async def crawl_website(url: str, max_pages: int = 25, max_depth: int = 3, concurrency: int = 5) -> Dict[str, Any]:
    """Helper function to create and run a crawler instance."""
    crawler = WebCrawler()
    await crawler.setup()
    try:
        return await crawler.crawl(url, max_pages, max_depth, concurrency)
    finally:
        await crawler.cleanup()

# --- Main Execution Block --- (Update print statements)
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
         print("Usage: python -m src.crawler.crawler <url> [options]")
         print("Running with default example: python -m src.crawler.crawler https://www.zaio.io/ -p 15 -d 2 -c 5")
         # --- Example Run --- #
         example_url = "https://www.zaio.io/" # Corrected example URL
         example_max_pages = 15
         example_max_depth = 2
         example_concurrency = 5
         BLOCK_RESOURCES = True # Use blocking for example
         # --- End Example Run --- #
         results = asyncio.run(crawl_website(example_url, example_max_pages, example_max_depth, example_concurrency))

    else:
         args = parser.parse_args()
         BLOCK_RESOURCES = args.block_resources
         print(f"Starting crawl: URL={args.url}, Max Pages={args.max_pages}, Max Depth={args.max_depth}, Concurrency={args.concurrency}, Resource Blocking={BLOCK_RESOURCES}")
         results = asyncio.run(crawl_website(args.url, args.max_pages, args.max_depth, args.concurrency))


    # Optional: Print summary after crawl
    if results and "crawl_metadata" in results:
        print("\n--- Crawl Metadata ---")
        meta = results['crawl_metadata']
        print(f"Start URL: {meta.get('start_url', 'N/A')}")
        print(f"Base Domain: {meta.get('base_domain', 'N/A')}")
        print(f"Tasks Completed: {meta.get('total_pages_processed', 'N/A')}")
        print(f"Unique URLs Added to Visited: {meta.get('total_urls_visited', 'N/A')}")
        print(f"Successful Analyses: {meta.get('successful_pages_analyzed', 'N/A')}")
        print(f"Failed/Error Pages: {meta.get('failed_or_error_pages', 'N/A')}")
        print(f"Total Time: {meta.get('total_crawl_time_seconds', 'N/A'):.2f} seconds")
        print(f"Final Queue Size: {meta.get('final_queue_size', 'N/A')}")
        print(f"Results File: {meta.get('results_file', 'N/A')}")
        print("-----------------------------")
        profile = results.get('business_profile', {})
        print("\n--- Aggregated Business Profile (Summary) ---")
        print(f"Business Name: {profile.get('business_name', 'N/A')}")
        print(f"Tagline: {profile.get('tagline_slogan', 'N/A')}")
        print(f"Description: {profile.get('short_description', 'N/A')}")
        print(f"Summary: {profile.get('aggregated_company_summary', 'N/A')}")
        print(f"Products Found: {len(profile.get('products', []))}")
        print(f"Services Found: {len(profile.get('services', []))}")
        print(f"Pricing Tiers Found: {len(profile.get('pricing_tiers', []))}")
        print(f"Testimonials Found: {len(profile.get('customer_testimonials', []))}")
        print(f"Case Studies Found: {len(profile.get('case_studies', []))}")
        print(f"FAQs Found: {len(profile.get('faqs', []))}")
        print(f"Social Links: {profile.get('social_media_links', {})}")
        print(f"Source URLs Processed: {len(profile.get('source_urls', []))}")
        print(f"(Full details in results JSON: {meta.get('results_file', 'N/A')})")
        print("-------------------------------------------")
    elif results and "error" in results:
         print(f"\nCrawl failed: {results['error']}")
    elif results is None:
         print("\nCrawl did not run (likely due to missing command-line arguments).")