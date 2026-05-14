"""Docker sandbox tool for the deep-agent stack.

Wraps executor.py's persistent-container engine as a LangChain tool.

Container lifecycle (managed by executor.py):
  - First call  → docker run  (fresh container, ~2 s)
  - Subsequent  → docker unpause if idle, otherwise already running (~100 ms)
  - After 30 min idle → watchdog pauses container (zero CPU, RAM preserved)

Pre-installed packages inside the container: numpy, pandas, scipy, sympy
(defined in the Dockerfile — add more there and rebuild with `docker build`).
"""

from typing import Annotated

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState
from tools.executor import run_all_users
from utils.state import DeepAgentState

CODE_SANDBOX_DESCRIPTION = """Execute Python code inside a persistent, isolated Docker sandbox.

## When to use
- Perform calculations, numerical work, or data transformations
- Validate logic or algorithms before presenting results
- Process or reformat structured data (JSON, CSV, etc.)
- Produce matplotlib / seaborn plots — they are captured and displayed automatically
- Any task where running real code is more reliable than reasoning about it

## Pre-installed packages
numpy, pandas, scipy, sympy, matplotlib, seaborn, scikit-learn
(import them freely — no installation step needed)

## Accessing uploaded files
All files the user has uploaded are copied into the sandbox's working directory.
Reference them by their original filename — no path prefix needed:

```python
import pandas as pd
df = pd.read_csv("Housing.csv")   # just the filename
print(df.head())
```

Use ls_files() to discover which files are present if unsure:
```python
import os
print(os.listdir("."))
```

## Plotting
Call plt.show() as usual — figures are captured, saved, and rendered in the UI automatically.
Always call plt.tight_layout() before plt.show() for clean output.

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

## Example — bar chart from uploaded CSV
```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("Housing.csv")
avg_price = df.groupby("furnishingstatus")["price"].mean().reset_index()

plt.figure(figsize=(8, 5))
sns.barplot(data=avg_price, x="furnishingstatus", y="price", palette="viridis")
plt.title("Average Price by Furnishing Status")
plt.tight_layout()
plt.show()
```
"""


@tool(description=CODE_SANDBOX_DESCRIPTION)
def run_code_in_sandbox(code: str, state: Annotated[DeepAgentState, InjectedState] = None) -> str:
    """Run Python code in the persistent Docker sandbox and return the output.

    Each call:
      1. Wakes the container if it was paused by the watchdog
      2. Copies any files from the virtual filesystem to the sandbox
      3. Writes code to a unique temp directory inside the container
      4. Runs it in parallel OS processes (one per configured USERS in executor.py)
      5. Deletes the temp directory after all processes finish
      6. Returns the combined stdout/stderr

    Args:
        code: Self-contained Python source code. Use print() to produce output.
        state: Agent state containing virtual filesystem (injected in tool node)

    Returns:
        Combined output from all user processes.
        Failures are prefixed with "ERROR:" and include the traceback.
    """
    # Get files from virtual filesystem if state is available
    files = state.get("files", {}) if state else {}
    
    result = run_all_users(code, files=files if files else None)

    # Surface errors clearly so the agent knows to fix and retry
    if "ERROR:" in result:
        return (
            f"Execution completed with errors:\n\n{result}\n\n"
            "Fix the code and call run_code_in_sandbox again."
        )

    return f"Execution successful:\n\n{result}"