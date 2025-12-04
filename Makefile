.PHONY: build_package
build_package:
	@rm -rf build dist *.egg-info; \
	uv run --no-cache python -m build

.PHONY: upload_package
upload_package:
	@uv publish --no-cache --token $(token)

.PHONY: publish 
publish:
	@make sync; \
	make build_package; \
	make upload_package token=$(shell cat ~/.ssh/.pypi-vecoli)

.PHONY: sync
sync: 
	@uv cache clean; \
	rm -f uv.lock; \
	uv lock --no-cache; \
	uv sync --no-cache --all-groups