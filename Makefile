pip-package:
	python setup.py bdist_wheel --universal

publish: pip-package
	scp dist/khepri-0.1.1-py2.py3-none-any.whl stukov:/srv/http/blog/

test:
	python -m unittest discover test/
