# Refactor Notes

+ The top goal of this refactor is to transform v2e to a set of usable API with extensible modules.
+ Only logging the following levels: INFO, WARNING, DEBUG. Instead of logging errors, we simply raise the error and exit the program.
+ Instead of using `argparse`, we use `questionary` for a guided configuration process.
+ Making emulator extensible via a registry.
+ Enforcing pre-commit checks.
+ Writing up developer guide.
+ The master branch should be protected, and any improvement should go through code review.
+ Future plan: clear benchmark and demo examples.
+ Parallelize for-loops via joblib.
+ Testing

## Precommit Hooks

Install precommit hooks

```
pre-commit install && pre-commit install -t pre-push
```

The pre-commit hooks will be checked during the push stage, and the code will only be pushed
if all the hooks are checked out. You can also manually run it via:

```
pre-commit run --all-files
```

Running pre-commit hooks can also be used as a code fixer.
