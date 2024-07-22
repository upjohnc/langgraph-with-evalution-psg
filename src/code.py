from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langsmith import Client
from langsmith.evaluation import evaluate

import constants
from lang_graph import run_graph

dataset_name = constants.DATASET_NAME


def create_chat_prompt():
    return ChatPromptTemplate.from_template(
        """
SYSTEM

You are a teacher grading a quiz.


You will be given a QUESTION, the GROUND TRUTH (correct) ANSWER, and the STUDENT ANSWER.


Here is the grade criteria to follow:

(1) Grade the student answers based ONLY on their factual accuracy relative to the ground truth answer.

(2) Ensure that the student answer does not contain any conflicting statements.

(3) It is OK if the student answer contains more information than the ground truth answer, as long as it is factually accurate relative to the  ground truth answer.


Score:

A score of 1 means that the student's answer meets all of the criteria. This is the highest (best) score.

A score of 0 means that the student's answer does not meet all of the criteria. This is the lowest possible score you can give.


Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct.


Avoid simply stating the correct answer at the outset.

HUMAN

QUESTION: {query}

GROUND TRUTH ANSWER: {correct_answer}

STUDENT ANSWER: {student_answer}

Grade the quiz based upon the above criteria with a score and your explanation for the score.
Provide the response in a JSON with a key 'score' for the score and a key 'explanation' for the explanation.
Respond only with the JSON repsonse.

"""
    )


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
            *[({"query": text}, {"output": label}) for text, label in examples]
        )
        _ = client.create_examples(
            inputs=inputs, outputs=outputs, dataset_id=dataset.id
        )


def answer_evaluator(run, example) -> dict:
    input_question = example.inputs["query"]
    reference = example.outputs["output"]
    prediction = run.outputs

    llm = ChatOllama(model=constants.MODEL)
    answer_grader = create_chat_prompt() | llm | JsonOutputParser()

    response = answer_grader.invoke(
        {
            "query": input_question,
            "correct_answer": reference,
            "student_answer": prediction,
        }
    )
    return {"score": response["score"], "key": "answer_v_reference_score"}


def main():

    client = Client()
    _ = create_dataset(client)

    experiment_prefix = "langgraph-evaluation"
    experiment_results = evaluate(
        run_graph,
        data=dataset_name,
        evaluators=[answer_evaluator],
        experiment_prefix=f"{experiment_prefix}-answer-and-tool-use",
        max_concurrency=1,
    )

    # query = "what's the frequency kenneth?"
    # query = "what is deltalake?"
    # response = run_graph(query)
    # llm = ChatOllama(model=constants.MODEL)
    # answer_grader = create_chat_prompt() | llm | JsonOutputParser()
    # response = answer_grader.invoke(
    #     {
    #         "question": query,
    #         # "correct_answer": "Song by REM",
    #         "correct_answer": "Deltalake refers to the Rust API (delta-rs) or Python API (also delta-rs) of Delta Lake,",
    #         "student_answer": response,
    #     }
    # )
    # return {"score": response["score"], "key": "answer_v_reference_score"}


if __name__ == "__main__":
    # query = "what's the frequency kenneth?"
    # if len(sys.argv) > 1 and sys.argv[1] == "deltalake":
    # query = "what is deltalake?"

    print(main())
# Provide the score:

# Explain your reasoning for the score:


# Give a binary score 1 or 0 score to indicate whether the document is relevant to the question. \n
# Provide the binary score as a  and no premable or explanation.
