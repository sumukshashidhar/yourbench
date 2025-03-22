.PHONY: style format quality check all

# Applies code style fixes to the specified file or directory
style:
	@echo "Applying style fixes to $(file)"
	ruff format $(file)
	ruff check --fix $(file)

# Checks code quality for the specified file or directory without applying fixes
check:
	@echo "Checking code quality for $(file) without fixes"
	ruff format --diff $(file)
	ruff check $(file)

# Applies PEP8 formatting and checks the entire codebase
all: style
	@echo "Formatting and checking the entire codebase"
	$(MAKE) style file=.

# Checks the entire codebase without applying fixes (for CI)
quality: check
	@echo "Checking quality of the entire codebase without fixes"
	$(MAKE) check file=.
