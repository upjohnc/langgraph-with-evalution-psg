from langchain_core.prompts import ChatPromptTemplate


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

Grade the quiz based upon the above criteria with a score.
Provide the response in a JSON with a key 'score' for the score.
Respond only with the JSON repsonse.

"""
    )
