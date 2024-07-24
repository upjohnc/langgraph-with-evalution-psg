---
# You can also start simply with 'default'
theme: seriph
# random image from a curated Unsplash collection by Anthony
# like them? see https://unsplash.com/collections/94734566/slidev
# some information about your slides (markdown enabled)
title: PSG - Langchain
# apply unocss classes to the current slide
class: text-center
# https://sli.dev/features/drawing
drawings:
  persist: false
# slide transition: https://sli.dev/guide/animations.html#slide-transitions
transition: slide-left
# enable MDC Syntax: https://sli.dev/features/mdc
mdc: true
---

# Langchain Evaluation

Pinnacle Solutions Group

Using evaluation to unit test the llm application

---
---
# Evaluation

## Unit test for your application


Definition:<br>
Testing of application using a defined truth as comparison to response for grading.

Langsmith has tooling to run many query prompts through your application
and compare to an expected result (a reference) to the answer returned by the application.

---
---
# Evaluation

## Data Creation
- create dataset for testing against
- store in langsmith


```python {all|2-8|14-16}
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

```
---
---
# Evaluation LLM Call (1/2)
Prompt
```python {all|5-6|20-21}
def create_chat_prompt():
    return ChatPromptTemplate.from_template(
        """
         SYSTEM
         You are a teacher grading a quiz.
         You will be given a QUESTION, the GROUND TRUTH (correct) ANSWER, and the STUDENT ANSWER.
      ...
         Score:
         
         A score of 1 means that the student's answer meets all of the criteria. This is the highest (best) score.
         
         A score of 0 means that the student's answer does not meet all of the criteria. This is the lowest possible score you can give.
      ... 
         HUMAN
         QUESTION: {query}
         GROUND TRUTH ANSWER: {correct_answer}
         STUDENT ANSWER: {student_answer}
         
         Grade the quiz based upon the above criteria with a score.
         Provide the response in a JSON with a key 'score' for the score.
         Respond only with the JSON repsonse.
         
         """
    )
```

---
---
# Evaluation LLM Call (2/2)

```python {all|7-15}

def answer_evaluator(run, example) -> dict:
    input_question = example.inputs["input"]
    reference = example.outputs["output"]
    prediction = run.outputs

    llm = ChatOllama(model=constants.MODEL)
    answer_grader = utils.create_chat_prompt() | llm | JsonOutputParser()

    response = answer_grader.invoke(
        {
            "query": input_question,
            "correct_answer": reference,
            "student_answer": prediction,
        }
    )
    return {"score": response["score"], "key": "answer_v_reference_score"}
```

---
---

# Evaluation Call

```python {all|7-13|8|10}
def evaluator():
    experiment_prefix = "langgraph-evaluation"

    client = Client()
    _ = create_dataset(client)

    _ = evaluate(
        run_graph,
        data=dataset_name,
        evaluators=[answer_evaluator],
        experiment_prefix=f"{experiment_prefix}-answer-and-tool-use",
        max_concurrency=1,
    )
```
---
---
# Langsmith Evaluation

<br>


![image](files://./langsmith.png)