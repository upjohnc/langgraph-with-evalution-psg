from langchain_core.vectorstores import VectorStoreRetriever
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from complete_response import get_end_response
from docs_evaluate import grade_docs_for_tavily_search
from tavily_search import web_search
from vector_store import create_vector_store, get_split_docs


class GraphState(TypedDict):
    """
    State of lang graph

    Attributes:
        query: query
        generation: LLM generation
        search: whether to add search
        documents: list of documents
    """

    query: str
    generation: str
    retriever: VectorStoreRetriever
    web_search: bool
    documents: list[str]
    steps: list[str]


def get_vector_store(state):
    state["steps"].append("get_vector_store")
    vector_store = create_vector_store(get_split_docs())
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})
    return {"query": state["query"], "retriever": retriever, "steps": state["steps"]}


def check_doc_grade(state):
    state["steps"].append("check_doc_grade")
    (tavily_search, docs) = grade_docs_for_tavily_search(
        state["retriever"], state["query"]
    )
    return {
        "query": state["query"],
        "retriever": state["retriever"],
        "documents": docs,
        "web_search": tavily_search,
        "steps": state["steps"],
    }


def decide_to_generate(state):
    if state["web_search"] is True:
        return "search"
    return "generate"


def web_tavily_search(state):
    state["steps"].append("web_tavily_search")
    # need to set to string
    # somehow the evaluator does not pass query as a string
    web_docs = web_search(query=str(state["query"]))
    state["documents"].extend(web_docs)
    return {
        "query": state["query"],
        "retriever": state["retriever"],
        "documents": state["documents"],
        "steps": state["steps"],
    }


def generate(state):
    state["steps"].append("generate")
    response = get_end_response(state["query"], state["documents"])
    return {
        "query": state["query"],
        "retriever": state["retriever"],
        "documents": state["documents"],
        "generation": response,
        "steps": state["steps"],
    }


def run_graph(query: str) -> str:
    workflow = StateGraph(GraphState)
    workflow.add_node("vector_retriever", get_vector_store)
    workflow.add_node("doc_grade", check_doc_grade)
    workflow.add_node("tavily_search", web_tavily_search)
    workflow.add_node("generate", generate)

    workflow.add_edge(START, "vector_retriever")
    workflow.add_edge("vector_retriever", "doc_grade")

    workflow.add_conditional_edges(
        "doc_grade",
        decide_to_generate,
        {"search": "tavily_search", "generate": "generate"},
    )

    workflow.add_edge("tavily_search", "generate")
    workflow.add_edge("generate", END)

    custom_graph = workflow.compile()

    state_dict = custom_graph.invoke({"query": query, "steps": []})
    return state_dict["generation"]
