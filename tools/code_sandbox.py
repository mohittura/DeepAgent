"""Docker sandbox tool for the deep-agent stack.

Wraps executor.py's persistent-container engine as a LangChain tool.

Container lifecycle (managed by executor.py):
  - First call  → docker run  (fresh container, ~2 s)
  - Subsequent  → docker unpause if idle, otherwise already running (~100 ms)
  - After 30 min idle → watchdog pauses container (zero CPU, RAM preserved)

Pre-installed packages inside the container: numpy, pandas, scipy, sympy
(defined in the Dockerfile — add more there and rebuild with `docker build`).
"""

from langchain_core.tools import tool
from tools.executor import run_all_users

CODE_SANDBOX_DESCRIPTION = """Execute Python code inside a persistent, isolated Docker sandbox.

## When to use
- Perform calculations, numerical work, or data transformations
- Validate logic or algorithms before presenting results
- Process or reformat structured data (JSON, CSV, etc.)
- Any task where running real code is more reliable than reasoning about it

## Pre-installed packages
numpy, pandas, scipy, sympy
(import them freely — no installation step needed)

## Parameters
- code: A complete, self-contained Python script. Use print() for output.
        Multiline strings are fully supported.

## Returns
A string with each user's execution result and exit status.
Check for "ERROR:" prefix in the output to detect failures — fix and retry.

## Constraints
- No network access inside the sandbox
- Container is shared and persistent — each call gets its own isolated
  /app/tmp/<uuid>/ directory so there is no state leakage between calls
- Use print() for output — return values from functions are not captured

## Example
```python
code = \"\"\"
import numpy as np
arr = np.array([1, 2, 3, 4, 5])
print(f"Mean: {arr.mean()}, Std: {arr.std():.4f}")
\"\"\"
```
"""


@tool(description=CODE_SANDBOX_DESCRIPTION)
def run_code_in_sandbox(code: str) -> str:
    """Run Python code in the persistent Docker sandbox and return the output.

    Each call:
      1. Wakes the container if it was paused by the watchdog
      2. Writes code to a unique temp directory inside the container
      3. Runs it in parallel OS processes (one per configured USERS in executor.py)
      4. Deletes the temp directory after all processes finish
      5. Returns the combined stdout/stderr

    Args:
        code: Self-contained Python source code. Use print() to produce output.

    Returns:
        Combined output from all user processes.
        Failures are prefixed with "ERROR:" and include the traceback.
    """
    result = run_all_users(code)

    # Surface errors clearly so the agent knows to fix and retry
    if "ERROR:" in result:
        return (
            f"Execution completed with errors:\n\n{result}\n\n"
            "Fix the code and call run_code_in_sandbox again."
        )

    return f"Execution successful:\n\n{result}"