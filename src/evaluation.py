from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
from langsmith import Client
from langsmith.evaluation import evaluate
from langsmith.schemas import Example, Run

import constants
import utils
from lang_graph import run_graph

dataset_name = constants.DATASET_NAME


def create_dataset(client: Client):
    examples = [
        ("what's the frequency kenneth?", "Song by REM"),
        (
            "what is deltalake?",
            "Deltalake refers to the Rust API (delta-rs) or Python API (also delta-rs) of Delta Lake",
        ),
    ]
    if not client.has_dataset(dataset_name=dataset_name):
        dataset = client.create_dataset(dataset_name)
        inputs, outputs = zip(
            *[({"input": text}, {"output": label}) for text, label in examples]
        )
        _ = client.create_examples(
            inputs=inputs, outputs=outputs, dataset_id=dataset.id
        )


def answer_evaluator(run: Run, example: Example) -> dict:
    input_question = example.inputs["input"]
    reference = example.outputs["output"]
    prediction = run.outputs["response"]

    llm = ChatOllama(model=constants.MODEL)
    answer_grader = utils.create_chat_prompt() | llm | JsonOutputParser()

    response = answer_grader.invoke(
        {
            "query": input_question,
            "correct_answer": reference,
            "student_answer": prediction,
        }
    )
    return {
        "score": response["score"],
        "key": "answer_v_reference_score",
    }


expected_steps_1 = [
    "get_vector_store",
    "check_doc_grade",
    "generate",
]

expected_steps_2 = [
    "get_vector_store",
    "check_doc_grade",
    "web_tavily_search",
    "generate",
]


def check_steps(run: Run, example: Example) -> dict:
    tool_calls = run.outputs["steps"]
    score = 0
    if tool_calls == expected_steps_1 or tool_calls == expected_steps_2:
        score = 1

    return {
        "score": score,
        "key": "tool_calls_in_correct_order",
    }


def evaluator():
    experiment_prefix = "langgraph-evaluation"

    client = Client()
    _ = create_dataset(client)

    result = evaluate(
        run_graph,
        data=dataset_name,
        evaluators=[answer_evaluator, check_steps],
        experiment_prefix=f"{experiment_prefix}-answer-and-tool-use",
        max_concurrency=1,
    )

    return result
