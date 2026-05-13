import httpx
import os
import asyncio
import time
import xml.etree.ElementTree as ET
from typing import Dict, Any, List, Optional

BASE_URL = "https://export.arxiv.org/api/query"
ARXIV_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom"
}

_last_request_time = 0.0
_request_lock = asyncio.Lock()
MIN_REQUEST_INTERVAL = 3.0

async def search_paper(query: str, 
                       max_results: Optional[int] = None, 
                       sort_by: str = 'submittedDate', 
                       order_by: str = 'descending',
                       client: Optional[httpx.AsyncClient] = None
                       ) -> List[Dict[str, Any]]:
    """
    Searches for papers on arXiv using the Atom API.
    """
    params = {
        "search_query": query,
        "sortBy": sort_by,
        "sortOrder": order_by
    }
    if max_results:
        params["max_results"] = max_results # type: ignore

    if client:
        response = await _rate_limit_request(client=client, url=BASE_URL, params=params)
        return _parse_arxiv_atom_response(response.text)
    else:
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as new_client:
            response = await _rate_limit_request(client=new_client, url=BASE_URL, params=params)
            return _parse_arxiv_atom_response(response.text)
    
async def _rate_limit_request(client: httpx.AsyncClient, url: str, params: Optional[Dict] = None) -> httpx.Response:
    global _last_request_time

    # Enforce minimum interval before sending to comply with arXiv's 3s rule
    async with _request_lock:
        elapsed = time.monotonic() - _last_request_time
        if elapsed < MIN_REQUEST_INTERVAL:
            await asyncio.sleep(MIN_REQUEST_INTERVAL - elapsed)
        _last_request_time = time.monotonic()

    for attempt in range(3): # Retry on timeout or 503
        try:
            response = await client.get(url, params=params)
            if response.status_code == 429:
                print("arXiv is rate limiting (429). Waiting 60 seconds...")
                await asyncio.sleep(60.0)
                continue
            if response.status_code == 503:
                print("arXiv service unavailable (503). Waiting 5 seconds...")
                await asyncio.sleep(5.0)
                continue
            
            response.raise_for_status()
            return response
        except (httpx.TimeoutException, httpx.NetworkError) as e:
            if attempt < 2:
                print(f"arXiv request failed ({e}). Retrying in 5s...")
                await asyncio.sleep(5.0)
            else:
                raise

    raise RuntimeError("arXiv request failed after retries.")
    
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
        Downloads the PDF asynchronously.
        """
        if not pdf_url:
            raise ValueError("No PDF URL available")
            
        if not filename:
            filename = pdf_url.split('/')[-1]
            if not filename.endswith(".pdf"):
                filename += ".pdf"
                
        path = os.path.join(dirpath, filename)

        if client:
            response = await client.get(pdf_url)
            response.raise_for_status()
            with open(path, 'wb') as f:
                f.write(response.content)
        else:
            async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as new_client:
                response = await new_client.get(pdf_url)
                response.raise_for_status()
                with open(path, 'wb') as f:
                    f.write(response.content)

        return path
