from typing import Any

from pydantic import BaseModel

from evaluation import evaluator


class EvaluationAssessment(BaseModel):
    evaluation_name: str
    score_succes_value: Any = 1
    evaluation_assertion: str | None = None

    def model_post_init(self, _):
        if self.evaluation_assertion is None:
            self.evaluation_assertion = self.evaluation_name


def test_evaluator():
    result = evaluator()

    all_evaluations = [
        EvaluationAssessment(
            evaluation_name="tool_calls_in_correct_order",
            score_succes_value=1,
        ),
        EvaluationAssessment(
            evaluation_name="answer_v_reference_score",
            evaluation_assertion="Answer v Reference is wrong",
        ),
    ]
    _ = assess_evaluator(result, all_evaluations)


def assess_evaluator(
    result, evaluation_assessments: list[EvaluationAssessment] | None = None
):
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
        assert all(
            [
                i == evaluation.score_succes_value
                for i in all_evaluations[evaluation.evaluation_name]
            ]
        ), evaluation.evaluation_assertion
