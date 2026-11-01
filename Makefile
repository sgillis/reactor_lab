.DEFAULT_GOAL := all

PDF_SOURCE := docs/model-summary.md
PDF_STYLE := docs/model-summary.css
PDF_OUTPUT := docs/model-summary.pdf
PDF_HTML := artifacts/model-summary.html

.PHONY: all sync test typecheck example surrogate demo pdf platform platform-down openapi frontend-generate

all: test typecheck example

sync:
	uv sync --all-groups

test: sync
	uv run pytest

typecheck: sync
	uv run basedpyright .

example: sync
	uv run reactor-lab simulate --plot

surrogate: sync
	uv run reactor-lab train --train-samples 512 --epochs 300

demo: sync
	uv run reactor-lab run-demo

platform:
	docker compose up --build

platform-down:
	docker compose down

openapi:
	uv run python -c 'import json; from pathlib import Path; from reactor_lab.platform.api import app; Path("openapi.json").write_text(json.dumps(app.openapi(), indent=2) + "\n", encoding="utf-8")'

frontend-generate: openapi
	cd frontend && npm run generate:api

pdf: $(PDF_OUTPUT)

$(PDF_OUTPUT): $(PDF_SOURCE) $(PDF_STYLE)
	mkdir -p artifacts
	pandoc $(PDF_SOURCE) --standalone --math-method=mathml --css=$(PDF_STYLE) --embed-resources --output=$(PDF_HTML)
	chromium --headless --no-sandbox --disable-gpu --no-pdf-header-footer --print-to-pdf=$(PDF_OUTPUT) file://$(CURDIR)/$(PDF_HTML)
