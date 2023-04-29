# Refactor Notes

+ Only logging the following levels: INFO, WARNING, DEBUG. Instead of logging errors, we simply raise the error and exit the program.
+ Instead of using `argparse`, we use `questionary` for a guided configuration process.
+ Making emulator extensible via a registry.
+ Enforcing pre-commit checks.
+ Writing up developer guide.
+ The master branch should be protected, and any improvement should go through code review.
+ Future plan: clear benchmark and demo examples.
+ Parallelize for-loops via joblib.
