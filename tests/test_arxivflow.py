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
