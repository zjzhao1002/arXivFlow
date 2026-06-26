import asyncio
import datetime
import ollama
from arxivflow import arXivFlow

def compose_email(title: str, authors: str, abstract: str) -> str:
    if not title or not authors or not abstract:
        return "No title, authors or abstract."
    
    model = "llama3.2"
    
    prompt = f"""
    You are an editor of Example Scientific Journal. You task is to outreach scientists and attract them to submit articles to your journal.
    {authors} have recently submitted an arXiv article called {title}. The abstract of this article is: \n\n
    {abstract}\n\n
    Compose an outreach email based on the given information. 
    You must show you are interested in their work and your journal would be the best choice for them.
    """

    response = ollama.chat(
        model = model,
        messages=[{"role": "user", "content": prompt}]
    )

    print(f"Composed email for title: {title}")

    return response['message']['content']

async def main():
    flow = arXivFlow(
        categories=['hep-ph'],
        ollama_model="llama3.2",
        max_results=10,
        start_date=datetime.datetime.now() - datetime.timedelta(days=7),
        end_date=datetime.datetime.now()
    )

    arxiv_data = await flow.get_arxiv_data(download_pdfs=True)
    if arxiv_data.empty:
        output_data = arxiv_data.reindex(
            columns=['arXiv ID','Authors', 'Emails', 'Title', 'Outreach Emails']
        )
        output_data.to_csv("output.csv", index=False)
        await flow.close()
        return

    emails = [
        await asyncio.to_thread(
            compose_email, 
            row['Title'],
            row['Authors'], 
            row['Abstract']
        )
        for _, row in arxiv_data.iterrows()
    ]

    arxiv_data['Outreach Emails'] = emails

    output_data = arxiv_data[['arXiv ID','Authors', 'Emails', 'Title', 'Outreach Emails']]

    output_data.to_csv("output.csv", index=False)

    await flow.close()

if __name__ == "__main__":
    asyncio.run(main())
