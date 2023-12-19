PYTHON='/home/ec2-user/anaconda3/envs/python3.10/bin/python'

pkg:
	-rm dist/*
	${PYTHON} -m build

pypi-test:
	${PYTHON} -m twine upload --repository-url https://test.pypi.org/legacy/ dist/*

pypi:
	${PYTHON} -m twine upload --verbose dist/*

changelog:
	f1=`mktemp`; \
	f2=`mktemp`; \
	git tag --sort=-committerdate | tee "$$f1" | sed -e 1d > "$$f2"; \
	paste "$$f1" "$$f2" | sed -e 's|	|...|g' | while read range; do echo; echo "## $$range"; git log '--pretty=format:* %s' "$$range"; done; \
	rm "$$f1" "$$f2"
	
clean:
	-rm -rf *.egg-info dist build xingu/__pycache__ *dist-info *pyproject-* .pyproject* .package_note*
