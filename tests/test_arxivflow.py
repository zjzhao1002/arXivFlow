import os
_parent_dp = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
_src_dp = os.path.join(_parent_dp, 'src')
import sys
sys.path.insert(0, _src_dp)

from arxivflow import arxivflow
import shutil

def test_set_client_parameters():
    """
    Test that set_client_parameters correctly updates the underlying arxiv.Client object.
    """
    # Initialize the arXivFlow class
    categories = ["cs.AI"]
    arxiv_flow = arxivflow.arXivFlow(categories=categories)
    
    # Set custom client parameters
    page_size = 50
    delay_seconds = 5.0
    num_retries = 5
    arxiv_flow.set_client_parameters(page_size=page_size, delay_seconds=delay_seconds, num_retries=num_retries)
    
    # Assert that the parameters are correctly set in the client object
    assert arxiv_flow.client.page_size == page_size
    assert arxiv_flow.client.delay_seconds == delay_seconds
    assert arxiv_flow.client.num_retries == num_retries

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
