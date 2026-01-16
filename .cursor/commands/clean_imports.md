# Clean Unused Imports

Remove unused imports from all Python files in the project.

## Instructions

1. **Identify Files**: Find all Python files in the project:
   ```bash
   find src -name "*.py" -type f
   ```

2. **Detect Unused Imports**: For each file, analyze imports and identify those not referenced in the code.

3. **Remove Unused Imports**: Edit each file to remove only the unused imports while:
   - Preserving all used imports
   - Maintaining import grouping (stdlib, third-party, local)
   - Keeping any `# noqa` or type-checking imports intact

4. **Verify**: After changes, run linter to confirm no new errors:
   ```bash
   uv run ruff check src/
   ```

## Import Categories to Check

- Standard library imports (`os`, `sys`, `datetime`, etc.)
- Third-party packages (`pandas`, `aiohttp`, `pydantic`, etc.)
- Local project imports (`from polybot.config import ...`)
- Type-only imports (`from typing import ...`, `TYPE_CHECKING` blocks)

## Constraints

- Do NOT remove imports used only in type hints
- Do NOT remove imports inside `if TYPE_CHECKING:` blocks
- Do NOT remove `__all__` exports from `__init__.py` files
- Do NOT reorder or reformat imports beyond removal
- PRESERVE imports with `# noqa` comments
- SKIP `__init__.py` files that re-export modules

## Examples

**Before:**
```python
import os
import sys
from datetime import datetime, timedelta
from typing import Optional

def get_time() -> datetime:
    return datetime.now()
```

**After:**
```python
from datetime import datetime

def get_time() -> datetime:
    return datetime.now()
```

