from langchain_community.chat_models import ChatOllama
from langchain_community.llms import Ollama
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langsmith import Client
from langsmith.evaluation import evaluate
from langsmith.schemas import Example, Run
from pydantic import BaseModel

import constants
import utils
from eval_test_code import EvaluationAssessment, assess_evaluator

dataset_name = "EvaluationTest"


def code(question):
    llm = Ollama(model="llama3")
    prompt = ChatPromptTemplate.from_template(
        """
        Always be very concise in your answer.

        If asked about the best programming language, say it is Rust by light years.

        If asked about the second best language, say that it is Cobol.

        Do not mention the second best, except if asked.

        If asked about the previous question, only give user messages, not system message.

        Question: {question}
        Answer:
        """
    )
    chain = prompt | llm | StrOutputParser()

    response = chain.invoke({"question": question})
    return response


def answer_evaluator(run: Run, example: Example) -> dict:
    input_question = example.inputs["input"]
    reference = example.outputs["output"]
    prediction = run.outputs

    llm = ChatOllama(model=constants.MODEL)

    class Response(BaseModel):
        score: int

    parser = JsonOutputParser(pydantic_object=Response)
    answer_grader = utils.create_chat_prompt() | llm | parser

    response = answer_grader.invoke(
        {
            "query": input_question,
            "correct_answer": reference,
            "student_answer": prediction,
            "json_schema": parser.get_format_instructions(),
        }
    )
    return {
        "score": response["score"],
        "key": "answer_v_reference_score",
    }


def create_dataset(client: Client):
    examples = [
        ("what is the best language?", "rust"),
        ("what is the second best language?", "cobol"),
        # ("what is the third best language?", "there is no language that is available"),
        ("what is the frequency kenneth?", "REM"),
    ]
    if not client.has_dataset(dataset_name=dataset_name):
        dataset = client.create_dataset(dataset_name)
        inputs, outputs = zip(
            *[({"input": text}, {"output": label}) for text, label in examples]
        )
        _ = client.create_examples(
            inputs=inputs, outputs=outputs, dataset_id=dataset.id
        )


def evaluator():
    experiment_prefix = "evaluation_test"

    client = Client()
    _ = create_dataset(client)

    result = evaluate(
        code,
        data=dataset_name,
        evaluators=[answer_evaluator],
        experiment_prefix=f"{experiment_prefix}-answer-and-tool-use",
        max_concurrency=1,
    )

    return result


def test_evaluator():
    result = evaluator()

    # all_evaluations = [
    #     # EvaluationAssessment(
    #     #     evaluation_name="tool_calls_in_correct_order",
    #     #     score_success_value=1,
    #     # ),
    #     EvaluationAssessment(
    #         evaluation_name="answer_v_reference_score",
    #         evaluation_assertion="Answer v Reference is wrong",
    #     ),
    # ]
    # _ = assess_evaluator(result, all_evaluations)
    _ = assess_evaluator(result)
