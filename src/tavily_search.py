from langchain.schema import Document
from langchain_community.tools.tavily_search import TavilySearchResults


def web_search(query: str) -> list[Document]:
    web_search_tool = TavilySearchResults(max_results=2)

    response = web_search_tool.invoke({"query": query})
    documents = [
        Document(page_content=doc["content"], metadata={"url": doc["url"]})
        for doc in response
    ]

    return documents
