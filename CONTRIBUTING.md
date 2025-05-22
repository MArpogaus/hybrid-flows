<!-- This file has been heavily inspired by https://github.com/probabilists/zuko/blob/master/CONTRIBUTING.md -->

# Contributing guidelines

First off, thank you for taking the time to contribute! 🎉

This document is a set of guidelines for contributing to this package, which includes how to ask questions, report issues, suggest enhancements, contribute code, etc.

## I just have a question

Please **don't file an issue** to ask a question. We use GitHub discussions as community forum for people to ask questions, share ideas or seek help. Before submitting your question, check whether it has already been asked in the discussions. If it has but the answer does not satisfy you, add a comment to the existing discussion instead of opening a new one.

## Submit an issue

Bugs and enhancements are tracked as GitHub issues. For common issues, such as bug reports and feature requests, templates are provided. It is strongly recommended to use them as it helps understand and resolve issues faster. A clear and concise title (e.g. "RuntimeError with X when Y") also helps other users and developers to find relevant issues.

Before submitting any issue, please perform a thorough search to see if your problem or a similar one has already been reported. If it has and the issue is still open, add a comment to the existing issue instead of opening a new one. If you only find closed issues related to your problem, open a new one and include links to the closed issues in the description.

## Contribute code

If you like the project and wish to contribute, you can start by looking at issues labeled `good first issue` (should only require a few lines of code) or `help wanted` (more involved). If you found a bug and want to fix it, please create an issue reporting the bug before creating a pull request. Similarly, if you want to add a new feature, first create a feature request issue. This allows to separate the discussions related to the bug/feature, from the discussions related to the fix/implementation.

To get started with contributing code, we recommend to install the package in editable mode with its development dependencies.

```
pip install -e ".[develop]"
```

Optionally, we also provide [pre-commit hooks](#pre-commit-hooks) to ensure that the code you commit adheres to our conventions.

```
pre-commit install
```

After installation, pre-commit will automatically execute the configured hooks before each commit and provide instructions on how to fix detected issues.

### Testing

We use [pytest](https://docs.pytest.org) to test our code base. If your contribution introduces new components, you should write new tests to make sure your code doesn't crash under normal circumstances. After installing `pytest`, add the tests to the [tests/](tests) directory and run them with

```
pytest tests
```

When you submit a pull request, tests are automatically (upon approval) executed for several versions of Python.

### Code conventions

We use [Ruff](https://github.com/astral-sh/ruff) to lint and format all Python code. After installing `ruff`, you can check if your code follows our conventions with

```
ruff check .
ruff format --check .
```

### Pre-Commit Hooks

To make it as easy as possible to comply with the conventions defined above, we offer pre-configured hooks for the [pre-commit framework](https://pre-commit.com/).
These hooks serve to ensure that all commits strictly comprise syntactically accurate and well-formatted code, and prevent potential issues such as accidental inclusion of large files.

Here's a quick overview of the hook used in the provided [configuration file](./.pre-commit-config.yaml) and their respective repository links:

- [ruff](https://github.com/astral-sh/ruff): This hook checks for formatting and linting errors in Python files.

- [pre-commit-hooks](https://github.com/pre-commit/pre-commit-hooks): These hooks validate YAML and TOML files, detect trailing whitespace, identify accidentally added large files, and pinpoint merge conflicts.

You can manually run pre-commit on the entire repository by using the command:

```
pre-commit run --all-files
```

This command is useful if you want to check all files in your repository without making a commit.

### Commits

We kindly ask to follow the [conventional commits specification](https://www.conventionalcommits.org/) for your commit messages.

Please make sure your description is short informative and dos not exceed the limit of =68= characters in the many git UIs.
For the same reason the body text line lengths should be limited to 80 characters.
