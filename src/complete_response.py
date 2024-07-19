from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser

import constants


def get_end_response(question, documents):
    prompt = PromptTemplate(
        template="""You are an assistant for question-answering tasks.

    Use the following documents to answer the question.

    If you don't know the answer, just say that you don't know.

    Use three sentences maximum and keep the answer concise:
    Question: {question}
    Documents: {documents}
    Answer:
    """,
        input_variables=["question", "documents"],
    )

    llm = ChatOllama(model=constants.MODEL, temperature=0)

    rag_chain = prompt | llm | StrOutputParser()
    return rag_chain.invoke({"question": question, "documents": documents})
