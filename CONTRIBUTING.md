# Contributing to Handwritten Character Recognition

Thank you for your interest in contributing! This document provides guidelines for contributing to this project.

## 🤝 How to Contribute

### Reporting Bugs

If you find a bug, please open an issue with:
- Clear description of the bug
- Steps to reproduce
- Expected vs actual behavior
- Screenshots (if applicable)
- Environment details (OS, Python version, etc.)

### Suggesting Features

Feature suggestions are welcome! Please open an issue describing:
- The feature and its benefits
- Potential implementation approach
- Any relevant examples or references

### Code Contributions

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make your changes**
   - Follow existing code style
   - Add tests if applicable
   - Update documentation
4. **Commit your changes**
   ```bash
   git commit -m "feat: Add your feature description"
   ```
5. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```
6. **Open a Pull Request**

## 📋 Development Setup

```bash
# Clone the repository
git clone https://github.com/dkumi12/CodeAlpha_HandwrittenRecognitionModel.git
cd CodeAlpha_HandwrittenRecognitionModel

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run tests
python test_api.py
```

## 🧪 Testing

- All new features should include tests
- Run the test suite before submitting PR
- Ensure all tests pass

## 📝 Commit Message Convention

We follow conventional commits:
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `test:` Test additions or changes
- `refactor:` Code refactoring
- `style:` Formatting changes
- `perf:` Performance improvements

## 💡 Areas for Contribution

### High Priority
- [ ] Add more training data for confused character pairs (O/0, I/1)
- [ ] Implement real-time drawing feedback
- [ ] Add data augmentation during inference
- [ ] Performance optimization for mobile devices

### Medium Priority
- [ ] Multi-language support
- [ ] Batch prediction endpoint
- [ ] Model explainability (GradCAM, LIME)
- [ ] A/B testing framework

### Documentation
- [ ] Add video tutorials
- [ ] Create API usage examples
- [ ] Write troubleshooting guide
- [ ] Add performance benchmarks

## 🎨 Code Style

- Follow PEP 8 for Python code
- Use type hints where appropriate
- Add docstrings to functions and classes
- Keep functions focused and modular
- Maximum line length: 100 characters

## 📞 Questions?

Feel free to open an issue for any questions or reach out via:
- LinkedIn: [David Osei Kumi](https://www.linkedin.com/in/daniel-kumi-9b5834205/)
- GitHub: [@dkumi12](https://github.com/dkumi12)

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.
