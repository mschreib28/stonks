# Atomic Commit Assistant

Create atomic, well-organized git commits for all currently uncommitted changes.

## Instructions

1. **Analyze Changes**: Run `git diff` and `git status` to understand all modifications.

2. **Group Logically**: Organize changes into atomic commits where each commit:
   - Addresses a single concern or feature
   - Is self-contained and could be reverted independently
   - Keeps related file changes together

3. **Commit Message Format**: Use conventional commit style:
   ```
   <type>(<scope>): <description>
   
   <optional body with details>
   ```
   Types: `feat`, `fix`, `refactor`, `chore`, `docs`, `perf`, `test`

4. **Execution**: Stage and commit each group separately using:
   ```bash
   git add <files>
   git commit -m "<message>"
   ```

5. **Order**: Commit foundational changes before dependent ones.

## Examples of Good Atomic Commits

- `feat(historical): add Kraken exchange support for OHLC data`
- `refactor(config): adjust signal thresholds for conservative trading`
- `perf(simulator): tune risk parameters for fractional Kelly sizing`

## Constraints

- Do NOT combine unrelated changes
- Do NOT create overly broad commits like "update multiple files"
- DO ask for confirmation before committing if changes are ambiguous
