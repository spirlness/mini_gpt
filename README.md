# mini_gpt

A minimal GPT implementation for character-level text generation.

[![CI](https://github.com/spirlness/mini_gpt/workflows/CI/badge.svg)](https://github.com/spirlness/mini_gpt/actions/workflows/ci.yml)

## Installation

```bash
pip install -e .
```

## Usage

Run training with default parameters:

```bash
python train.py
```

Or with custom parameters:

```bash
python train.py --data_path data/your_data.txt --batch_size 64 --lr 0.001 --max_steps 5000
```

You can also use it as a module:

```bash
python -m mini_gpt.train --help
```

## Development

### Setup

1. Clone the repository:
```bash
git clone https://github.com/spirlness/mini_gpt.git
cd mini_gpt
```

2. Install in development mode:
```bash
pip install -e .
```

3. Install pre-commit hooks:
```bash
pip install pre-commit
pre-commit install
```

### Linting

This project uses [Ruff](https://github.com/astral-sh/ruff) for linting and formatting:

```bash
# Install ruff
pip install ruff

# Check code
ruff check .

# Format code
ruff format .
```

## CI/CD

This project uses GitHub Actions for continuous integration and deployment:

### Continuous Integration (CI)

The CI pipeline runs on every push and pull request to the `main`/`master` branches:

- **Lint**: Checks code quality using Ruff linter and formatter
- **Test**: Tests package installation on Python 3.9, 3.10, and 3.11
- **Build**: Builds the Python package and uploads artifacts

### Continuous Deployment (CD)

The CD pipeline automatically deploys to PyPI when a new release is published:

- Builds the package
- Publishes to PyPI using trusted publishing

## License

MIT
