.PHONY: rmd-to-pdf # compile rmd to pdf
rmd-to-pdf:
	Rscript -e 'for(p in c("rmarkdown", "ISLR", "IRkernel")) if(!requireNamespace(p, quietly = TRUE)) install.packages(p, repos = "https://cran.rstudio.com")'
	Rscript -e "rmarkdown::render('$(filepath)', output_format = 'pdf_document')"
	rm -rf *.bib *.aux *.log *.out *.synctex.gz

.PHONY: md-to-pdf # compile md to pdf
md-to-pdf:
	pandoc "$(filepath)" -o "$(basename $(filepath)).pdf"

.PHONY: up # pull and push changes
up:
	git pull
	git add .
	if [ -z "$(msg)" ]; then git commit -m "up"; else git commit -m "$(msg)"; fi
	git push

.PHONY: help # generate help message
help:
	@echo "Usage: make [target]\n"
	@grep '^.PHONY: .* #' makefile | sed 's/\.PHONY: \(.*\) # \(.*\)/\1	\2/' | expand -t20
