from typing import Any

import pytest_check
from langsmith.evaluation._runner import ExperimentResults
from pydantic import BaseModel


class EvaluationAssessment(BaseModel):
    """
    Attributes of the individual to be used is the parsing of the evaluation

    Attributes:
        evaluation_name (str): the name of the evaluation from the callable passed to the `evaluate` function
        score_success_value (Any): the value that the grader llm returns when the evaluation is correct
        evaluation_assertion (str | None): the string that should be sent when the assertion fails, defaults to the evaluation_name
    """

    evaluation_name: str
    score_success_value: Any = 1
    evaluation_assertion: str | None = None

    def model_post_init(self, _):
        if self.evaluation_assertion is None:
            self.evaluation_assertion = self.evaluation_name


def assess_evaluator(
    result: ExperimentResults,
    evaluation_assessments: list[EvaluationAssessment] | None = None,
):
    """Assert that each of the evaluations in a langchain evaluation return correct score.

    Parses the evaluation object that is returned from the langchain `evaluation` function
    and then asserts that all of the tests for an evaluation has the correct score.

    It assumes that the evaluation has the key of `score` from the grader llm response.

    If no evaluation_assessments are passed then it simply uses the evaulation keys that
    are set in the evaluation function.  Also the default score is `1`.

    Args:
        result (ExperimentResults): object that is returned from the langchain evaluate function
        evaluation_assessments (list[EvaluationAssessment] | None): definition of the individual evaluations in the test run
    """
    all_results = []
    for i in result._results:
        for j in i["evaluation_results"]["results"]:
            all_results.append(j)
    evaluation_names = set(i.key for i in all_results)

    if evaluation_assessments is None:
        evaluation_assessments = [
            EvaluationAssessment(evaluation_name=i) for i in evaluation_names
        ]

    all_evaluations = {}
    for j in evaluation_names:

        results = [i.score for i in all_results if i.key == j]
        all_evaluations[j] = results

    for evaluation in evaluation_assessments:
        with pytest_check.check:
            assert all(
                [
                    i == evaluation.score_success_value
                    for i in all_evaluations[evaluation.evaluation_name]
                ]
            ), evaluation.evaluation_assertion
