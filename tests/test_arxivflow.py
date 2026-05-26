import os
_parent_dp = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
_src_dp = os.path.join(_parent_dp, 'src')
import sys
sys.path.insert(0, _src_dp)

from arxivflow import arxivflow
import shutil
import pytest
import httpx

@pytest.mark.anyio
async def test_client_initialization():
    """
    Test that the arXivFlow class correctly initializes the httpx AsyncClient.
    """
    # Initialize the arXivFlow class
    categories = ["cs.AI"]
    arxiv_flow = arxivflow.arXivFlow(categories=categories)
    
    # Assert that the client is an httpx.AsyncClient
    assert isinstance(arxiv_flow.client, httpx.AsyncClient)
    assert arxiv_flow.client.is_closed is False
    
    # Close the client
    await arxiv_flow.close()
    assert arxiv_flow.client.is_closed is True

def test_set_pdfs_path():
    """
    Test that set_pdfs_path correctly updates the path and creates the directory.
    """
    # Initialize the arXivFlow class
    categories = ["cs.AI"]
    arxiv_flow = arxivflow.arXivFlow(categories=categories)
    
    # Set a custom PDF path
    custom_path = "test_pdfs_dir_setters"
    if os.path.exists(custom_path):
        shutil.rmtree(custom_path)
        
    arxiv_flow.set_pdfs_path(custom_path)
    
    # Assert that the path is set correctly and the directory is created
    assert arxiv_flow.pdfs_path == custom_path
    assert os.path.exists(custom_path)
    assert os.path.isdir(custom_path)
    
    # Cleanup
    shutil.rmtree(custom_path)

@pytest.mark.anyio
async def test_get_arxiv_data_replaces_previous_results(monkeypatch):
    async def fake_search_paper(query, max_results, client):
        return [
            {
                "arXiv ID": query.split(" ")[0].replace("cat:", ""),
                "Title": "Test title",
                "Authors": "Test Author",
                "Abstract": "Test abstract",
                "Categories": "cs.AI",
                "Published Date": "2026-05-26",
                "Updated Date": "2026-05-26",
                "arXiv URL": "https://arxiv.org/abs/1234.5678",
                "PDF URL": "https://arxiv.org/pdf/1234.5678",
            }
        ]

    monkeypatch.setattr(arxivflow, "search_paper", fake_search_paper)

    arxiv_flow = arxivflow.arXivFlow(categories=["cs.AI", "cs.LG"], max_results=10)
    first_df = await arxiv_flow.get_arxiv_data()
    second_df = await arxiv_flow.get_arxiv_data()

    assert len(first_df) == 2
    assert len(second_df) == 2
    assert len(arxiv_flow.dfs) == 2

    await arxiv_flow.close()

@pytest.mark.anyio
async def test_max_results_per_category_uses_ceiling_and_minimum():
    arxiv_flow = arxivflow.arXivFlow(categories=["cs.AI", "cs.LG"], max_results=1)
    assert arxiv_flow._max_results_per_category() == 1
    await arxiv_flow.close()

    arxiv_flow = arxivflow.arXivFlow(categories=["cs.AI", "cs.LG", "cs.CV"], max_results=10)
    assert arxiv_flow._max_results_per_category() == 4
    await arxiv_flow.close()

    arxiv_flow = arxivflow.arXivFlow(categories=["cs.AI"], max_results=None)
    assert arxiv_flow._max_results_per_category() is None
    await arxiv_flow.close()
