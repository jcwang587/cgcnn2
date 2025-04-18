# Contributing

We welcome contributions to CGCNN2! Here's how you can help:

## How to Contribute

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Development Setup

1. Clone the repository:
```bash
git clone https://github.com/jcwang587/cgcnn2.git
cd cgcnn2
```

2. Install development dependencies:
```bash
pip install -e ".[dev]"
```

3. Install pre-commit hooks:
```bash
pre-commit install
```

## Code Style

We follow PEP 8 style guidelines. Please ensure your code follows these guidelines before submitting a pull request.

## Testing

Please ensure all tests pass before submitting a pull request:
```bash
pytest
```

## Documentation

When adding new features, please update the documentation accordingly.

## Issue Reporting

If you find a bug or have a feature request, please open an issue on GitHub. 