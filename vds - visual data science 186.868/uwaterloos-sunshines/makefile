# --------------------------------------------------------------- venv

.PHONY: venv # infer dependencies from code, initialize venv
venv:
	pip install pip --upgrade
	pip install pipreqs
	rm -rf requirements.txt requirements.in
	pipreqs . --mode no-pin --encoding utf-8 --ignore .venv
	mv requirements.txt requirements.in

	pip install pip-tools
	pip-compile requirements.in -o requirements.txt -vvv
	
	rm -rf .venv
	python -m venv .venv
	./.venv/bin/python3 -m pip install -r requirements.txt
	@echo "to activate venv, run: source .venv/bin/activate"

.PHONY: venv-lock # freeze venv into requirements.txt
venv-lock:
	./.venv/bin/python3 -m pip freeze > requirements.in

.PHONY: in-lock # compile requirements.in
in-lock:
	pip-compile requirements.in -o requirements.txt -vvv

# --------------------------------------------------------------- docker

.PHONY: docker-up # run docker container
docker-up:
	docker-compose up --detach
	@echo "to exec into docker container, run: docker exec -it main bash"

.PHONY: docker-build # save changes to container
docker-build:
	docker-compose build

.PHONY: docker-clean # wipe everything in all docker containers
docker-clean:
	docker-compose down

	docker stop $$(docker ps -a -q) || true
	docker rm $$(docker ps -a -q) || true
	docker rmi $$(docker images -q) || true
	yes | docker container prune
	yes | docker image prune
	yes | docker volume prune
	yes | docker network prune
	yes | docker system prune
	
	docker ps --all
	docker images
	docker system df
	docker volume ls
	docker network ls

# --------------------------------------------------------------- conda

.PHONY: conda-reqs-to-yaml # install conda to generate environment.yml from requirements.txt (idempotent)
conda-reqs-to-yaml:
	conda update -n base -c defaults conda
	conda config --env --set subdir osx-arm64 || true
	conda config --set auto_activate_base false
	conda info
	bash -c '\
		source $$(conda info --base)/etc/profile.d/conda.sh && conda activate base; \
		conda create --yes --name con python=3.11; \
		source $$(conda info --base)/etc/profile.d/conda.sh && conda activate con; \
		pip install -r requirements.txt; \
		conda env export --no-builds | grep -v "prefix:" > environment.yml; \
		source $$(conda info --base)/etc/profile.d/conda.sh; conda deactivate; \
		conda remove --yes --name con --all; \
	'

.PHONY: conda # install conda from environment.yml file
conda:
	# can also be used in docker with continuumio/miniconda3 image
	bash -c '\
		source $$(conda info --base)/etc/profile.d/conda.sh; conda activate base; \
		conda env create --file environment.yml --solver=libmamba; \
	'
	@echo "to activate conda environment, run: conda activate con"

.PHONY: conda-clean # wipe conda environment
conda-clean:
	# conda clean --all # wipe everything
	bash -c '\
		source $$(conda info --base)/etc/profile.d/conda.sh; conda activate base; \
		conda remove --yes --name con --all; \
		source $$(conda info --base)/etc/profile.d/conda.sh; conda deactivate; \
	'

# --------------------------------------------------------------- nohup

.PHONY: monitor # create nohup with restart on failure
monitor:
	if [ "$(filepath)" = "" ]; then echo "missing 'filepath' argument"; exit 1; fi
	bash -c '\
		monitor() { \
			while true; do \
				if ! ps -p $$(cat "monitor-process.pid" 2>/dev/null) > /dev/null 2>&1; then \
					echo "$$(date): process not running or died, (re)starting..." >> monitor.log; \
					nohup python "$(filepath)" > "monitor-process.log" 2>&1 & \
					echo $$! > "monitor-process.pid"; \
					echo "$$(date): started process with PID $$(cat monitor-process.pid)" >> monitor.log; \
				fi; \
				sleep 5; \
			done; \
		}; \
		monitor >> "monitor.log" 2>&1 & \
		echo $$! > "monitor.pid"; \
		echo "$$(date): monitor started" >> "monitor.log"; \
	'

.PHONY: monitor-watch # tail log of nohup process
monitor-watch:
	while true; do clear; tail -n 100 monitor-process.log; sleep 0.1; done
	# watch -n 0.1 "tail -n 100 monitor-process.log"

.PHONY: monitor-kill # kill nohup process
monitor-kill:
	kill -9 $$(cat monitor.pid) || true
	rm -rf monitor.pid
	rm -rf monitor.log
	kill -9 $$(cat monitor-process.pid) || true
	rm -rf monitor-process.pid
	rm -rf monitor-process.log

# --------------------------------------------------------------- utils

.PHONY: rmd-to-pdf # compile rmd to pdf
rmd-to-pdf:
	Rscript -e 'for(p in c("rmarkdown", "ISLR", "IRkernel")) if(!requireNamespace(p, quietly = TRUE)) install.packages(p, repos = "https://cran.rstudio.com")'
	Rscript -e "rmarkdown::render('$(filepath)', output_format = 'pdf_document')"
	rm -rf *.bib *.aux *.log *.out *.synctex.gz

.PHONY: md-to-pdf # compile md to pdf
md-to-pdf:
	pandoc "$(filepath)" -o "$(basename $(filepath)).pdf"

.PHONY: fmt # format codebase
fmt:
	./.venv/bin/python3 -m pip install isort
	./.venv/bin/python3 -m pip install ruff
	./.venv/bin/python3 -m pip install autoflake

	./.venv/bin/python3 -m isort .
	./.venv/bin/python3 -m autoflake --remove-all-unused-imports --recursive --in-place .
	./.venv/bin/python3 -m ruff format --config line-length=500 .

.PHONY: sec # check for vulns
sec:
	./.venv/bin/python3 -m pip install bandit
	./.venv/bin/python3 -m pip install safety
	
	./.venv/bin/python3 -m bandit -r .
	./.venv/bin/python3 -m safety check --full-report

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
