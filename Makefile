T := NULL
init:
	pip3 install -r requirements.txt

test:
	@# Usage:
	@# If running specific test:
	@# make test T=NameOfTestFunction
	@# If running all tests:
	@# make test
	@if [ $T = 'NULL' ]; then \
		SKIP=1 python3 -m unittest tests.test_burgersDA.TestBlockMethods; \
	else \
		SKIP=0 python3 -m unittest tests.test_burgersDA.TestBlockMethods.$T; \
	fi
