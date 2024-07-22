import sys

from lang_graph import run_graph

if __name__ == "__main__":
    query = "what's the frequency kenneth?"
    if len(sys.argv) > 1 and sys.argv[1] == "deltalake":
        query = "what is deltalake?"

    print(run_graph(query))
