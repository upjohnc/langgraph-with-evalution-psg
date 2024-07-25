import sys

from evaluation import evaluator
from lang_graph import run_graph


def run_evaluator():
    _ = evaluator()


def run_search(query: str) -> str:
    return run_graph(query)


if __name__ == "__main__":
    cli_arg = None if len(sys.argv) <= 1 else sys.argv[1]
    if cli_arg == "search":
        print(run_search("what is deltalake?"))
    else:
        run_evaluator()
