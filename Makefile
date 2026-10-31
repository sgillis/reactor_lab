.DEFAULT_GOAL := all

PDF_SOURCE := docs/model-summary.md
PDF_STYLE := docs/model-summary.css
PDF_OUTPUT := docs/model-summary.pdf
PDF_HTML := artifacts/model-summary.html

.PHONY: all sync test example pdf

all: test example

sync:
	uv sync

test: sync
	uv run pytest

example: sync
	uv run python examples/plot_trajectories.py

pdf: $(PDF_OUTPUT)

$(PDF_OUTPUT): $(PDF_SOURCE) $(PDF_STYLE)
	mkdir -p artifacts
	pandoc $(PDF_SOURCE) --standalone --math-method=mathml --css=$(PDF_STYLE) --embed-resources --output=$(PDF_HTML)
	chromium --headless --no-sandbox --disable-gpu --no-pdf-header-footer --print-to-pdf=$(PDF_OUTPUT) file://$(CURDIR)/$(PDF_HTML)
