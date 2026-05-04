.PHONY: preview render clean

preview:
	quarto preview

render:
	quarto render

clean:
	rm -rf _site .quarto
