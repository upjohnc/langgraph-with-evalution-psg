from langchain.schema import Document
from langchain_community.llms import Ollama
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

import constants


def grade_docs_for_tavily_search(
    retriever, question: str
) -> tuple[bool, list[Document]]:
    """
    Grade docs for relevancy to query.

    If all docs are relevant than do not need additionally search through Tavily


    Args:
        retriever : vector store retriever
        question (str): query question from user

    Returns:
        bool: True if need Tavily search needed
    """
    docs = retriever.invoke(question)

    model = Ollama(model=constants.MODEL)
    prompt = ChatPromptTemplate.from_template(
        """You are a teacher grading a quiz. You will be given:
        1/ a QUESTION
        2/ A FACT provided by the student

        You are grading RELEVANCE RECALL:
        A score of 1 means that ANY of the statements in the FACT are relevant to the QUESTION.
        A score of 0 means that NONE of the statements in the FACT are relevant to the QUESTION.
        1 is the highest (best) score. 0 is the lowest score you can give.

        Avoid simply stating the correct answer at the outset.

        Question: {question} \n
        Fact: \n\n {documents} \n\n

        Give a binary score 1 or 0 score to indicate whether the document is relevant to the question. \n
        Provide the binary score as a JSON with a single key 'score' and no premable or explanation.
        """
    )
    retrieval_grader = prompt | model | JsonOutputParser()

    def check_more_search(doc: Document) -> bool:
        doc_text = doc.page_content
        response = retrieval_grader.invoke(
            {"question": question, "documents": doc_text}
        )
        return response["score"] == 1

    relevant_docs = [doc for doc in docs if check_more_search(doc) is True]
    more_search = len(relevant_docs) != len(docs)
    return more_search, relevant_docs
