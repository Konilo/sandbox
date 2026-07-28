.PHONY: preview render clean

preview:
	uv run quarto preview

render:
	uv run quarto render

clean:
	rm -rf _site .quarto
