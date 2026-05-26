import httpx
import pytest

from arxivflow import arxiv_functions


def _atom_feed(*ids):
    entries = "\n".join(
        f"""
        <entry>
          <id>http://arxiv.org/abs/{paper_id}v1</id>
          <title>Paper {paper_id}</title>
          <summary>Abstract {paper_id}</summary>
          <author><name>Author {paper_id}</name></author>
          <published>2026-05-26T00:00:00Z</published>
          <updated>2026-05-26T00:00:00Z</updated>
          <arxiv:primary_category term="cs.AI" />
          <link title="pdf" href="https://arxiv.org/pdf/{paper_id}" />
        </entry>
        """
        for paper_id in ids
    )
    return f"""
    <feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
      {entries}
    </feed>
    """


@pytest.mark.anyio
async def test_search_paper_fetches_paged_results(monkeypatch):
    monkeypatch.setattr(arxiv_functions, "MIN_REQUEST_INTERVAL", 0.0)
    monkeypatch.setattr(arxiv_functions, "_last_request_time", 0.0)
    monkeypatch.setattr(arxiv_functions, "_cooldown_until", 0.0)
    starts = []

    def handler(request):
        start = int(request.url.params.get("start", 0))
        starts.append(start)
        pages = {
            0: _atom_feed("1001.0001", "1001.0002"),
            2: _atom_feed("1001.0003", "1001.0004"),
            4: _atom_feed("1001.0005"),
        }
        return httpx.Response(200, text=pages[start])

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        results = await arxiv_functions.search_paper(
            query="cat:cs.AI",
            max_results=5,
            client=client,
            page_size=2,
        )

    assert starts == [0, 2, 4]
    assert [paper["arXiv ID"] for paper in results] == [
        "1001.0001",
        "1001.0002",
        "1001.0003",
        "1001.0004",
        "1001.0005",
    ]


@pytest.mark.anyio
async def test_rate_limit_request_retries_retryable_status(monkeypatch):
    monkeypatch.setattr(arxiv_functions, "MIN_REQUEST_INTERVAL", 0.0)
    monkeypatch.setattr(arxiv_functions, "_last_request_time", 0.0)
    monkeypatch.setattr(arxiv_functions, "_cooldown_until", 0.0)
    monkeypatch.setattr(arxiv_functions, "_retry_delay", lambda attempt, response=None: 0.0)
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(503)
        return httpx.Response(200, text=_atom_feed("1001.0001"))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        response = await arxiv_functions._rate_limit_request(client, "https://export.arxiv.org/api/query")

    assert response.status_code == 200
    assert calls == 2


@pytest.mark.anyio
async def test_download_pdf_writes_atomically(monkeypatch, tmp_path):
    monkeypatch.setattr(arxiv_functions, "MIN_REQUEST_INTERVAL", 0.0)
    monkeypatch.setattr(arxiv_functions, "_last_request_time", 0.0)
    monkeypatch.setattr(arxiv_functions, "_cooldown_until", 0.0)

    def handler(request):
        return httpx.Response(200, content=b"%PDF-1.7\ncontent")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        path = await arxiv_functions.download_pdf(
            pdf_url="https://arxiv.org/pdf/1001.0001",
            dirpath=str(tmp_path),
            filename="paper.pdf",
            client=client,
        )

    assert (tmp_path / "paper.pdf").read_bytes() == b"%PDF-1.7\ncontent"
    assert not (tmp_path / "paper.pdf.part").exists()
    assert path == str(tmp_path / "paper.pdf")


def test_retry_delay_uses_long_cooldown_for_429(monkeypatch):
    monkeypatch.setattr(arxiv_functions.random, "uniform", lambda start, end: 0.0)
    response = httpx.Response(429)

    assert arxiv_functions._retry_delay(0, response) == arxiv_functions.RATE_LIMIT_COOLDOWN
    assert arxiv_functions._retry_delay(1, response) == arxiv_functions.RATE_LIMIT_COOLDOWN * 2
