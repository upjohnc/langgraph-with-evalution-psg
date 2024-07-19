from docs_evaluate import grade_docs_for_tavily_search
from tavily_search import web_search
from vector_store import create_vector_store, get_split_docs


def main():
    # create vector store with rust deltalake docs
    vector_store = create_vector_store(get_split_docs())
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})

    # score retrieved documents
    # question = "what's the frequency kenneth?"
    question = "what is deltalake?"
    (tavily_search, docs) = grade_docs_for_tavily_search(retriever, question)

    # tavily search for web retrieval
    if tavily_search:
        web_docs = web_search(query=question)
        docs.extend(web_docs)

    # graph
    # retrieve docs
    # score docs
    # branch if score is bad
    # if bad tavily search
    # generate response from inputs


if __name__ == "__main__":
    main()
