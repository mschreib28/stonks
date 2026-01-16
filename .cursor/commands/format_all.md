# Format All Files

Format all code files in the project using standard formatters and any detected project configuration files.

## Overview

This command formats all Python and TypeScript (or JavaScript) source files in the repository, automatically applying the appropriate formatter (e.g., `black`, `prettier`, etc.). It ensures code style consistency by detecting and using any project-specific config files (such as `.black`, `pyproject.toml`, `.prettierrc`, etc.). If no configuration is found, it falls back to standard defaults.

---

## Step 1: Select TypeScript/JavaScript Formatter

The following formatters are commonly used for TypeScript/JavaScript projects. **Choose one formatter for this project**:

- `prettier` (Recommended; supports TypeScript, JavaScript, JSON, CSS, etc.)
- `eslint --fix` (For basic fixes according to ESLint rules; can supplement Prettier)

**Please reply with your formatter choice if using TypeScript/JavaScript files. For Python-only projects, skip this step.**

---

## Step 2: Detect Python Formatting Configuration

- If `.black`, `pyproject.toml` (with `[tool.black]` section), or `setup.cfg` (with `[black]` section) exists in the project root, use those for configuration.
- Otherwise, use the default `black` settings as fallback.

---

## Step 3: Detect TypeScript/JavaScript Formatting Configuration

- If `.prettierrc`, `.prettierrc.json`, `.prettierrc.js`, `.prettierrc.yaml`, `.prettierrc.yml`, `.prettierrc.toml`, `prettier.config.js`, or `package.json` (`prettier` key present) exist, use those for `prettier` settings.
- If using `eslint`, check for `.eslintrc`, `.eslintrc.js`, `.eslintrc.json`, or the `eslintConfig` key in `package.json`.
- Otherwise, use default settings.

---

## Step 4: Format All Files

### Python

```bash
# Find all Python files and format with black
find src -name "*.py" -type f | xargs black
```
- Add `--config <file>` to the `black` command if config detected.

### TypeScript/JavaScript (if selected)

#### Example with Prettier:

```bash
# Find and format all JS/TS/JSON/CSS/MD files
find src -type f \( -name "*.js" -o -name "*.ts" -o -name "*.jsx" -o -name "*.tsx" -o -name "*.json" -o -name "*.md" -o -name "*.css" \) \
  | xargs prettier --write
```
- Add `--config <file>` if config detected.

---

## Step 5: Verify Formatting

After formatting, run the relevant formatters in **check** mode to verify all files were properly formatted:

```bash
# For Python
black --check src/

# For TypeScript/JavaScript (if chosen)
prettier --check "src/**/*.{js,jsx,ts,tsx,json,css,md}"
```

---

## Notes & Best Practices

- Never mix unrelated file types into the same formatting command.
- Only format files tracked by git unless intentionally reformatting generated/untracked code.
- If the project uses **other languages**, extend this guide with language-appropriate formatters (e.g., `rustfmt` for Rust, `gofmt` for Go, etc.).
- Always check the output for unintentional bulk reformatting, especially on files with unusual encodings or large diffs.

---

## Example Workflow

```bash
# 1. Python formatting
find src -name "*.py" -type f | xargs black

# 2. TypeScript/JavaScript formatting (if selected)
find src -type f \( -name "*.js" -o -name "*.ts" \) | xargs prettier --write

# 3. Verify
black --check src/
prettier --check "src/**/*.{js,ts}"
```

---

## Confirm Formatter Choice

**If this is a TypeScript/JavaScript project, please confirm your desired formatter from the following before running formatting:**

- prettier
- eslint --fix
- other (please specify)

If only Python files are present, you may proceed with black formatting as described.
