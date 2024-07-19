from vector_store import create_vector_store, get_split_docs


def main():
    # create vector store with rust deltalake docs
    vector_store = create_vector_store(get_split_docs())

    # score retrieved documents

    # tavily search for web retrieval

    # graph
    # retrieve docs
    # score docs
    # branch if score is bad
    # if bad tavily search
    # generate response from inputs


if __name__ == "__main__":
    main()
