# Contributing

## Getting Started

1. Fork the repository
2. Create a feature branch: `git checkout -b my-feature`
3. Install dependencies: `pip install -r requirements.txt`

## Development Workflow

This project follows test-driven development:

1. **Write a failing test first** — describe the behaviour you want
2. **Write minimal code to pass** — no more than needed
3. **Refactor** — clean up with tests still passing

## Running Tests

```bash
pytest -v
pytest --cov=. --cov-report=term-missing
```

Coverage must stay above 90% for new code.

## Submitting a PR

- Keep changes focused — one feature or fix per PR
- Ensure all tests pass and coverage does not drop
- Write a clear PR description explaining what and why
