pytest --strict-markers \
       --mypy \
       --mypy-ignore-missing-imports \
       --cov=wav2rec \
       --cov-report term-missing \
       --cov-fail-under 90 \
       --ignore=experiments \
       --showlocals \
       -vv
