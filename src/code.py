from lang_evaluate import grade_docs_for_tavily_search
from vector_store import create_vector_store, get_split_docs


def main():
    # create vector store with rust deltalake docs
    vector_store = create_vector_store(get_split_docs())
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})

    # score retrieved documents
    question = "what's the frequency kenneth?"
    # question = "what is deltalake?"
    tavily_search = grade_docs_for_tavily_search(retriever, question)

    # tavily search for web retrieval

    # graph
    # retrieve docs
    # score docs
    # branch if score is bad
    # if bad tavily search
    # generate response from inputs


if __name__ == "__main__":
    main()
