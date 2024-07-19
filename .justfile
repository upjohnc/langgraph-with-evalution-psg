default:
    just --list

install:
    poetry install --sync --no-root --with dev

pre-commit:
    pre-commit install

run:
    PYTHONPATH=. poetry run python src/code.py

ollama-start:
    ollama serve

llama3:
    ollama pull llama3
