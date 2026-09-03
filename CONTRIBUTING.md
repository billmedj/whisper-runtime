# Contributing

Whisper Execution Runtime is a pre-alpha reference implementation. Changes must
keep the runtime contract small, testable, and independent of application
policy.

## Before you start

Open an issue before a large API, state-model, backend, or formal-model change.
A focused bug fix can go directly to a pull request.

Do not include model checkpoints, audio recordings, credentials, machine paths,
or generated build output.

## Set up the repository

```sh
python -m venv .venv
```

Activate `.venv` with the command for your operating system, then install the
package and validation tools:

```sh
python -m pip install -e ".[validation,quality]" "build>=1.2,<2"
```

## Validate a change

```sh
python -B -m unittest discover -s tests -v
python -B -m unittest discover -s tools -p "test_*.py" -v
python -m ruff check src tests tools examples
python -m ruff format --check src tests tools examples
python -m mypy src
python -B tools/check_repository.py
python -B examples/minimal_transaction.py
python -m build
python -B tools/check_distribution.py dist
```

Run `lake build` from `formal/lean` when a change affects the state model,
resource accounting, lifecycle, or published claims about the formal model.

## Change requirements

A pull request must state:

- the observed problem;
- the ownership or lifecycle rule involved;
- the compatibility effect;
- the tests and evidence added or changed.

Keep performance claims separate from correctness claims. Include the complete
environment and input identity for hardware measurements.

Changes copied or derived from third-party projects must retain their source,
copyright, and license notices. The OpenAI Whisper integration patches are
covered by the MIT terms described in `THIRD_PARTY_NOTICES.md`.

Unless stated otherwise, an original contribution submitted to this repository
is provided under the repository's Apache License 2.0 terms.
