# Contributing to Multi-Teacher LLM Distillation Engine

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Code of Conduct

Please be respectful and considerate in all interactions. We aim to maintain a welcoming environment for all contributors.

## How to Contribute

### Reporting Issues

- Check if the issue already exists in the GitHub Issues
- Provide a clear description with steps to reproduce
- Include your environment details (OS, Python version, GPU, etc.)
- Add relevant error messages and logs

### Suggesting Features

- Open an issue with the `enhancement` label
- Clearly describe the feature and its benefits
- Explain the use case and expected behavior

### Pull Requests

1. **Fork the repository** and create your branch from `main`
2. **Make your changes** following the code style guidelines
3. **Test your changes** thoroughly
4. **Update documentation** if needed
5. **Submit a pull request** with a clear description

## Development Setup

```bash
# Clone your fork
git clone https://github.com/your-username/multi_teacher_llm_distillation_engine.git
cd multi_teacher_llm_distillation_engine

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements.txt
pip install pytest black flake8

# Run tests
pytest tests/
```

## Code Style

- Follow PEP 8 guidelines
- Use meaningful variable and function names
- Add docstrings to all functions and classes
- Keep functions focused and concise
- Add type hints where appropriate

### Formatting

We use `black` for code formatting:

```bash
black src/
```

### Linting

We use `flake8` for linting:

```bash
flake8 src/ --max-line-length=100
```

## Testing

- Write tests for new features
- Ensure all tests pass before submitting PR
- Aim for good test coverage

```bash
pytest tests/ -v
```

## Documentation

- Update README.md for user-facing changes
- Add docstrings for new functions/classes
- Include examples for new features
- Update configuration documentation

## Commit Messages

- Use clear, descriptive commit messages
- Start with a verb (Add, Fix, Update, Remove, etc.)
- Keep the first line under 50 characters
- Add detailed description if needed

Example:
```
Add support for custom loss functions

- Implement CustomLossRegistry class
- Add configuration options for loss selection
- Update documentation with examples
```

## Areas for Contribution

### High Priority

- [ ] Distributed training support (multi-GPU, multi-node)
- [ ] TensorBoard integration
- [ ] Comprehensive test suite
- [ ] Performance optimizations
- [ ] Documentation improvements

### Feature Ideas

- [ ] Automatic hyperparameter tuning
- [ ] Model pruning and compression
- [ ] Additional student architectures
- [ ] Evaluation and benchmarking tools
- [ ] API server for inference
- [ ] Docker containers
- [ ] More loss function options

### Bug Fixes

Check the Issues page for bugs labeled `good first issue` or `help wanted`.

## Questions?

Feel free to open an issue with the `question` label or reach out to the maintainers.

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.

---

Thank you for contributing! 🎉
