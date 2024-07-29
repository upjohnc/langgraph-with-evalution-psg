default:
    just --list

install:
    poetry install --sync --no-root --with dev

pre-commit:
    pre-commit install

run *args:
    PYTHONPATH=./src poetry run python src/code.py {{ args }}

tests *args:
    PYTHONPATH=./src poetry run pytest . {{ args }} -v

ollama-start:
    ollama serve

llama3:
    ollama pull llama3
