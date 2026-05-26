import httpx
import os
import asyncio
import time
import random
import xml.etree.ElementTree as ET
from typing import Dict, Any, List, Optional

BASE_URL = "https://export.arxiv.org/api/query"
ARXIV_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom"
}

ARXIV_HEADERS = {
    "User-Agent": "arxivflow/0.2.2 (https://github.com/zjzhao1002/arXivFlow; research tool)"
}

_last_request_time = 0.0
_cooldown_until = 0.0
_request_lock = asyncio.Lock()
_request_semaphore = asyncio.Semaphore(1)
MIN_REQUEST_INTERVAL = 3.0
RATE_LIMIT_COOLDOWN = 60.0
DEFAULT_TIMEOUT = 60.0
MAX_RETRIES = 5
DEFAULT_PAGE_SIZE = 100
MAX_PAGE_SIZE = 2000
RETRY_STATUS_CODES = {429, 500, 502, 503, 504}

async def search_paper(query: str, 
                       max_results: Optional[int] = None, 
                       sort_by: str = 'submittedDate', 
                       order_by: str = 'descending',
                       client: Optional[httpx.AsyncClient] = None,
                       page_size: int = DEFAULT_PAGE_SIZE
                       ) -> List[Dict[str, Any]]:
    """
    Searches for papers on arXiv using the Atom API.
    """
    page_size = max(1, min(page_size, MAX_PAGE_SIZE))
    if client:
        return await _search_paper_with_client(
            query=query,
            max_results=max_results,
            sort_by=sort_by,
            order_by=order_by,
            client=client,
            page_size=page_size,
        )

    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT, follow_redirects=True) as new_client:
        return await _search_paper_with_client(
            query=query,
            max_results=max_results,
            sort_by=sort_by,
            order_by=order_by,
            client=new_client,
            page_size=page_size,
        )

async def _search_paper_with_client(
        query: str,
        max_results: Optional[int],
        sort_by: str,
        order_by: str,
        client: httpx.AsyncClient,
        page_size: int
    ) -> List[Dict[str, Any]]:
    results = []
    seen_ids = set()
    start = 0
    requested_total = max_results

    while True:
        current_page_size = page_size
        if requested_total is not None:
            remaining = requested_total - len(results)
            if remaining <= 0:
                break
            current_page_size = min(current_page_size, remaining)

        params = {
            "search_query": query,
            "sortBy": sort_by,
            "sortOrder": order_by,
            "start": start,
            "max_results": current_page_size,
        }

        response = await _rate_limit_request(client=client, url=BASE_URL, params=params)
        page_results = _parse_arxiv_atom_response(response.text)
        if not page_results:
            break

        for paper in page_results:
            arxiv_id = paper["arXiv ID"]
            if arxiv_id in seen_ids:
                continue
            seen_ids.add(arxiv_id)
            results.append(paper)

        if requested_total is None or len(page_results) < current_page_size:
            break
        start += current_page_size

    return results
    
async def _rate_limit_request(client: httpx.AsyncClient, url: str, params: Optional[Dict] = None) -> httpx.Response:
    global _last_request_time, _cooldown_until

    async with _request_semaphore:
        for attempt in range(MAX_RETRIES):
            # Enforce minimum interval before sending to comply with arXiv's 3s rule.
            async with _request_lock:
                cooldown_remaining = _cooldown_until - time.monotonic()
                if cooldown_remaining > 0:
                    await asyncio.sleep(cooldown_remaining)
                elapsed = time.monotonic() - _last_request_time
                if elapsed < MIN_REQUEST_INTERVAL:
                    await asyncio.sleep(MIN_REQUEST_INTERVAL - elapsed)
                _last_request_time = time.monotonic()

            try:
                response = await client.get(url, params=params, headers=ARXIV_HEADERS)
                if response.status_code in RETRY_STATUS_CODES and attempt < MAX_RETRIES - 1:
                    delay = _retry_delay(attempt, response)
                    if response.status_code == 429:
                        _cooldown_until = max(_cooldown_until, time.monotonic() + delay)
                    print(f"arXiv returned {response.status_code}. Retrying in {delay:.1f}s...")
                    await asyncio.sleep(delay)
                    continue
                
                response.raise_for_status()
                return response
            except (httpx.TimeoutException, httpx.NetworkError) as e:
                if attempt >= MAX_RETRIES - 1:
                    raise
                delay = _retry_delay(attempt)
                print(f"arXiv request failed ({e}). Retrying in {delay:.1f}s...")
                await asyncio.sleep(delay)

    raise RuntimeError("arXiv request failed after retries.")

def _retry_delay(attempt: int, response: Optional[httpx.Response] = None) -> float:
    if response is not None:
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            try:
                return max(float(retry_after), MIN_REQUEST_INTERVAL)
            except ValueError:
                pass
        if response.status_code == 429:
            return min(RATE_LIMIT_COOLDOWN * (2 ** attempt) + random.uniform(0.0, 1.0), 300.0)

    base_delay = MIN_REQUEST_INTERVAL * (2 ** attempt)
    jitter = random.uniform(0.0, 1.0)
    return min(base_delay + jitter, 60.0)
    
def _parse_arxiv_atom_response(text: str) -> List[Dict[str, Any]]:
    results = []

    try: 
        root = ET.fromstring(text)
        for entry in root.findall('atom:entry', ARXIV_NS):
            id_elem = entry.find('atom:id', ARXIV_NS)
            if id_elem is None or id_elem.text is None:
                continue
            
            arxiv_url = id_elem.text
            paper_id = arxiv_url.split('/abs/')[-1]
            short_id = paper_id.split('v')[0]

            title_elem = entry.find('atom:title', ARXIV_NS)
            title = (
                title_elem.text.strip().replace('\n', ' ') 
                if title_elem is not None and title_elem.text 
                else ""
                )
            
            authors = []
            for author in entry.findall('atom:author', ARXIV_NS):
                name_elem = author.find('atom:name', ARXIV_NS)
                if name_elem is not None and name_elem.text:
                    authors.append(name_elem.text)

            summary_elem = entry.find('atom:summary', ARXIV_NS)
            abstract = (
                summary_elem.text.strip().replace('\n', ' ')
                if summary_elem is not None and summary_elem.text
                else ""
            )

            categories = []
            for cat in entry.findall('arxiv:primary_category', ARXIV_NS):
                term = cat.get('term')
                if term:
                    categories.append(term)
            for cat in entry.findall('atom:category', ARXIV_NS):
                term = cat.get('term')
                if term and term not in categories:
                    categories.append(term)

            published = entry.findtext('atom:published', default="", namespaces=ARXIV_NS)[:10]
            updated = entry.findtext('atom:updated', default="", namespaces=ARXIV_NS)[:10]

            pdf_url = None
            for link in entry.findall('atom:link', ARXIV_NS):
                if link.get('title') == 'pdf':
                    pdf_url = link.get('href')
                    break
            if not pdf_url:
                pdf_url = f"https://arxiv.org/pdf/{paper_id}"

            results.append(
                {
                    "arXiv ID": short_id,
                    "Title": title,
                    "Authors": ", ".join(authors),
                    "Abstract": abstract,
                    "Categories": ", ".join(categories),
                    "Published Date": published,
                    "Updated Date": updated,
                    "arXiv URL": arxiv_url,
                    "PDF URL": pdf_url
                }
            )
    except ET.ParseError as e:
        raise ValueError(f"Failed parsing arXiv API response: {e}")
    
    return results

async def download_pdf(
        pdf_url: str,
        dirpath: str = "./",
        filename: str = "",
        client: Optional[httpx.AsyncClient] = None
    ) -> str:
        """
        Downloads the PDF asynchronously, respecting arXiv's rate limits.
        """
        if not pdf_url:
            raise ValueError("No PDF URL available")
            
        if not filename:
            filename = pdf_url.split('/')[-1]
            if not filename.endswith(".pdf"):
                filename += ".pdf"
                
        path = os.path.join(dirpath, filename)
        tmp_path = f"{path}.part"
        os.makedirs(dirpath, exist_ok=True)

        if client:
            response = await _rate_limit_request(client=client, url=pdf_url)
        else:
            async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT, follow_redirects=True) as new_client:
                response = await _rate_limit_request(client=new_client, url=pdf_url)

        if not response.content.startswith(b"%PDF"):
            raise ValueError(f"Downloaded content from {pdf_url} is not a PDF.")

        try:
            with open(tmp_path, 'wb') as f:
                f.write(response.content)
            os.replace(tmp_path, path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        return path
