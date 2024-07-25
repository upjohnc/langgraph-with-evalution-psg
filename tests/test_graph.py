from evaluation import evaluator


def test_evaluator():
    result = evaluator()
    all_results = []
    for i in result._results:
        for j in i["evaluation_results"]["results"]:
            all_results.append(j)

    score_tool_order = [
        i.score for i in all_results if i.key == "tool_calls_in_correct_order"
    ]
    score_answer_v_reference = [
        i.score for i in all_results if i.key == "answer_v_reference_score"
    ]

    # when score = 1 then the result is correct, and 1 is truthy
    assert all(score_tool_order), "Tool order is wrong"
    assert all(score_answer_v_reference), "Answer v Reference is wrong"
