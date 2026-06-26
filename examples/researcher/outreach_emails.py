import asyncio
import datetime
import ollama
from arxivflow import arXivFlow

def normalize(keyword: str):
    keyword = keyword.lower().strip()
    return keyword

def score_article(article_keywords: list[str], user_keywords: list[str]) -> float:
    if not article_keywords or not user_keywords:
        return 0
    article_set = set(normalize(k) for k in article_keywords)
    user_set = set(normalize(k) for k in user_keywords)

    matches = article_set & user_set
    return len(matches) / len(user_set)

def parse_keywords(keywords: str | list[str]) -> list[str]:
    if isinstance(keywords, list):
        return keywords
    if not isinstance(keywords, str):
        return []
    return [keyword.strip() for keyword in keywords.split(",") if keyword.strip()]

def compose_email(user_article: str, 
                  user_abstract: str, 
                  target_article: str,
                  target_authors: str, 
                  target_abstract: str) -> str:
    
    model = "llama3.2"

    prompt = f"""
    You are a physicist working in high energy physics. You find that {target_authors} have published an arXiv article with title "{target_article}".
    The abstract of this article is:\n\n
    {target_abstract}\n\n
    This article is highly relevant to your work: {user_article}. Compose an email to them to bring their attention to your work. 
    In this email, you must show your interest on their work and summarize your own work based on this abstract:\n\n
    {user_abstract}\n\n
    Finally, you can kindly remind them to cite your paper.
    """

    response = ollama.chat(
        model = model, 
        messages = [{"role": "user", "content": prompt}]
    )

    print(f"Composed email for title: {target_article}")

    return response['message']['content']

async def main():
    user_keywords = ["Dark Matter", "Collider", "Dark Energy"]
    user_article = "Searching for dark matter at colliders"
    user_abstract = "We explore the searches for dark matter and dark energy at LHC and future colliders."

    flow = arXivFlow(
        categories=['hep-ph'],
        ollama_model="llama3.2",
        max_results=20,
        start_date=datetime.datetime.now() - datetime.timedelta(days=7),
        end_date=datetime.datetime.now()
    )   

    raw_data = await flow.get_arxiv_data(download_pdfs=True)

    if raw_data.empty:
        output_data = raw_data.reindex(
            columns=['arXiv ID','Authors', 'Emails', 'Title', 'Relevant Score', 'Outreach Emails']
        )
        output_data.to_csv("output.csv", index=False)
        await flow.close()
        return

    raw_data['Relevant Score'] = raw_data['Keywords'].apply(
        lambda keywords: score_article(parse_keywords(keywords), user_keywords)
    )
    arxiv_data = raw_data[raw_data['Relevant Score'] > 0.0].copy()

    emails = [
        await asyncio.to_thread(
            compose_email, 
            user_article,
            user_abstract,
            row['Title'],
            row['Authors'], 
            row['Abstract']
        )
        for _, row in arxiv_data.iterrows()
    ]

    arxiv_data['Outreach Emails'] = emails
    output_data = arxiv_data[['arXiv ID', 'Authors', 'Emails', 'Title', 'Relevant Score', 'Outreach Emails']]
    output_data.to_csv("output.csv", index=False)

    await flow.close()

if __name__ == "__main__":
    asyncio.run(main())
