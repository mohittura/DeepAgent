import os
import time
import uuid
import threading
import subprocess
import base64
import glob
from multiprocessing import Process, Queue
from queue import Empty

# ── Config ────────────────────────────────────────────────────
CONTAINER_NAME    = "sandbox-merged"
DOCKER_IMAGE      = "sandbox-merged"
PYTHON_BIN        = "/.venv/bin/python3"
TMP_DIR           = "/app/tmp"
FILENAME          = "solution.py"
USERS             = ["A"]

INACTIVITY_LIMIT  = 30 * 60        # 30 minutes in seconds
WATCHDOG_INTERVAL = 60              # check every 60 seconds


# ─────────────────────────────────────────────────────────────
# Activity Tracker
# ─────────────────────────────────────────────────────────────

_last_activity = time.time()
_activity_lock = threading.Lock()


def _record_activity():
    """Call this on every incoming request."""
    global _last_activity
    with _activity_lock:
        _last_activity = time.time()


def _seconds_since_last_activity() -> float:
    with _activity_lock:
        return time.time() - _last_activity


# ─────────────────────────────────────────────────────────────
# Container State Helpers
# ─────────────────────────────────────────────────────────────

def _get_container_status() -> str:
    """
    Returns the container's current Docker status string.
    Possible values: 'running', 'paused', 'exited', 'created', or '' if not found.
    """
    result = subprocess.run(
        ["docker", "inspect", "--format", "{{.State.Status}}", CONTAINER_NAME],
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _container_is_running() -> bool:
    """True only when the container is fully running (not paused, not stopped)."""
    return _get_container_status() == "running"


# ─────────────────────────────────────────────────────────────
# Container Lifecycle
# ─────────────────────────────────────────────────────────────

def _start_container():
    """
    Brings the container to a running state based on its current status:
      - paused  → docker unpause  (~100ms, resumes exactly where it was frozen)
      - exited  → docker start    (~2s, only after a manual stop or host reboot)
      - missing → docker run      (first-ever launch)
    """
    status = _get_container_status()

    if status == "paused":
        print(f"  [Container] Unpausing '{CONTAINER_NAME}'...", flush=True)
        subprocess.run(
            ["docker", "unpause", CONTAINER_NAME],
            check=True,
            capture_output=True,
        )
        # unpause is synchronous — container is immediately ready, no sleep needed

    elif status == "exited":
        print(f"  [Container] Starting stopped container '{CONTAINER_NAME}'...", flush=True)
        subprocess.run(
            ["docker", "start", CONTAINER_NAME],
            check=True,
            capture_output=True,
        )
        time.sleep(2)   # give Docker time to fully initialize

    else:
        # Container doesn't exist at all — very first run ever
        print(f"  [Container] Running fresh container from image '{DOCKER_IMAGE}'...", flush=True)
        try:
            subprocess.run([
                "docker", "run",
                "-d",                       # detached
                "--name", CONTAINER_NAME,
                DOCKER_IMAGE,
                "sleep", "infinity",        # keep it alive indefinitely
            ], check=True, capture_output=True)
            time.sleep(2)
        except subprocess.CalledProcessError as e:
            error_msg = e.stderr.decode() if e.stderr else str(e)
            if "Cannot connect to the Docker daemon" in error_msg or "docker daemon" in error_msg.lower():
                raise RuntimeError(
                    "Docker daemon is not running. Please start Docker Desktop or the Docker daemon and try again."
                )
            elif "No such image" in error_msg or "image not found" in error_msg.lower():
                raise RuntimeError(
                    f"Docker image '{DOCKER_IMAGE}' not found. Please build it first with: "
                    f"docker build -t {DOCKER_IMAGE} ."
                )
            elif "already in use" in error_msg.lower():
                # Container name collision — try to remove and retry
                print(f"  [Container] Container name '{CONTAINER_NAME}' already in use, removing old container...", flush=True)
                subprocess.run(
                    ["docker", "rm", "-f", CONTAINER_NAME],
                    capture_output=True,
                )
                subprocess.run([
                    "docker", "run",
                    "-d",
                    "--name", CONTAINER_NAME,
                    DOCKER_IMAGE,
                    "sleep", "infinity",
                ], check=True, capture_output=True)
                time.sleep(2)
            else:
                raise RuntimeError(f"Failed to start Docker container: {error_msg}")

    # Ensure /app/tmp exists (important after a fresh container run)
    subprocess.run(
        ["docker", "exec", CONTAINER_NAME, "mkdir", "-p", TMP_DIR],
        capture_output=True,
    )

    print(f"  [Container] '{CONTAINER_NAME}' is ready.\n", flush=True)


def _pause_container():
    """
    Freezes the container via cgroups SIGSTOP.
      - CPU drops to zero immediately
      - RAM is still held (process state preserved)
      - Resume via unpause is ~100ms with no re-initialization
    """
    print(f"\n  [Watchdog] Pausing '{CONTAINER_NAME}' due to inactivity...", flush=True)
    subprocess.run(
        ["docker", "pause", CONTAINER_NAME],
        capture_output=True,
    )
    print(f"  [Watchdog] Container paused — zero CPU, instant resume on next request.\n", flush=True)


def _ensure_container_running():
    """
    Called before every request.
    If the container is paused or stopped, this wakes it up first.
    """
    if not _container_is_running():
        print(f"  [Container] Not running — waking up...", flush=True)
        try:
            _start_container()
        except RuntimeError as e:
            print(f"  [ERROR] {e}\n", flush=True)
            raise


# ─────────────────────────────────────────────────────────────
# Watchdog — background daemon thread
# ─────────────────────────────────────────────────────────────

def _watchdog_loop():
    """
    Runs every WATCHDOG_INTERVAL seconds for the lifetime of the agent.
    Pauses the container after INACTIVITY_LIMIT seconds of no requests.
    Does nothing if the container is already paused or stopped.
    """
    while True:
        time.sleep(WATCHDOG_INTERVAL)

        idle_seconds = _seconds_since_last_activity()

        if idle_seconds >= INACTIVITY_LIMIT:
            if _container_is_running():
                _pause_container()
            else:
                print(
                    f"  [Watchdog] {idle_seconds / 60:.1f} min idle — "
                    f"container already paused/stopped.",
                    flush=True,
                )
        else:
            remaining = (INACTIVITY_LIMIT - idle_seconds) / 60
            print(
                f"  [Watchdog] Active — {remaining:.1f} min until auto-pause if no requests.",
                flush=True,
            )


def start_watchdog():
    """
    Starts the watchdog as a daemon thread (dies automatically when agent exits).
    Called once at module import time — no manual setup needed.
    """
    thread = threading.Thread(target=_watchdog_loop, daemon=True)
    thread.start()
    print(
        f"  [Watchdog] Started — container will pause after "
        f"{INACTIVITY_LIMIT // 60} min of inactivity.\n",
        flush=True,
    )


def _wrap_code_with_plot_capture(code: str, request_dir: str) -> str:
    """
    Wrap user code with matplotlib backend setup and plot capture.

    Strategy: set MPLBACKEND=Agg via os.environ BEFORE any matplotlib import
    so the non-interactive backend is guaranteed regardless of import order.
    With the Agg backend plt.show() is a no-op (figures stay open), so we
    simply save every open figure in a footer after user code finishes.
    No monkey-patching required.
    """
    header = (
        'import os as _os\n'
        'import sys as _sys\n'
        '\n'
        '# Force the non-interactive Agg backend before matplotlib is imported.\n'
        '# Setting the env var here works even when user code does its own\n'
        '# "import matplotlib.pyplot as plt" — backend is locked in at import time.\n'
        '_os.environ["MPLBACKEND"] = "Agg"\n'
        f'_PLOT_DIR = "{request_dir}"\n'
        '\n'
        '# ── user code ────────────────────────────────────────────────\n'
    )

    footer = (
        '\n'
        '# ── plot capture (runs after user code) ─────────────────────\n'
        'try:\n'
        '    import matplotlib.pyplot as _plt\n'
        '    _open_figs = _plt.get_fignums()\n'
        '    for _idx, _fnum in enumerate(_open_figs, start=1):\n'
        '        _fig = _plt.figure(_fnum)\n'
        '        _plot_path = _os.path.join(_PLOT_DIR, f"plot_{_idx:03d}.png")\n'
        '        _fig.savefig(_plot_path, dpi=100, bbox_inches="tight")\n'
        '        print(f"[PLOT SAVED] {_plot_path}", flush=True)\n'
        '    if _open_figs:\n'
        '        _plt.close("all")\n'
        'except Exception as _exc:\n'
        '    print(f"[PLOT CAPTURE ERROR] {_exc}", flush=True)\n'
    )

    return header + code + footer

# ─────────────────────────────────────────────────────────────
# Private — collects plot files
# ─────────────────────────────────────────────────────────────

def _collect_plots(request_dir_in_container: str) -> str:
    """Collect all PNG plots from the request directory and return as base64 embedded images."""
    # Use docker exec to find PNG files in the container
    result = subprocess.run(
        ["docker", "exec", CONTAINER_NAME, "find", request_dir_in_container, "-name", "plot_*.png", "-type", "f"],
        capture_output=True,
        text=True,
    )
    
    plot_files = result.stdout.strip().split('\n') if result.stdout.strip() else []
    
    output_parts = []
    for plot_file in plot_files:
        if plot_file:
            # Read the PNG file from container and encode as base64
            result = subprocess.run(
                ["docker", "exec", CONTAINER_NAME, "base64", "-w", "0", plot_file],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                base64_data = result.stdout.strip()
                output_parts.append(f"[IMAGE]\n<img src='data:image/png;base64,{base64_data}' style='max-width:100%;' />\n[/IMAGE]")
    
    return "\n".join(output_parts) if output_parts else ""

def _run_user(user_label: str, filepath: str, request_dir: str, queue: Queue):
    """
    Each Process calls this independently.
    Runs the solution.py file and collects any generated plots.
    """
    try:
        # Extract the directory from filepath to set as working directory
        request_dir_only = str(os.path.dirname(filepath))
        filename_only = os.path.basename(filepath)
        
        result = subprocess.run(
            ["docker", "exec", "-w", request_dir_only, CONTAINER_NAME, PYTHON_BIN, filename_only],
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            output = result.stdout.strip() or "✓ (no output)"
            # Collect any generated plots
            plots = _collect_plots(request_dir)
            if plots:
                output = f"{output}\n\n{plots}"
        else:
            # Show both stdout and stderr so errors are never silently empty
            stdout = result.stdout.strip()
            stderr = result.stderr.strip()
            output = f"ERROR: {stderr or stdout or 'unknown error (no output captured)'}"
    except Exception as exc:
        output = f"ERROR: worker failed before returning output: {exc}"

    queue.put((user_label, output))
    print(f"  [User-{user_label}] Complete", flush=True)


def _collect_worker_results(processes: list[tuple[str, Process]], queue: Queue) -> list[str]:
    """
    Drain worker results before joining processes.

    Large plot payloads are sent through multiprocessing.Queue. If the parent
    joins first, a child can block while flushing the queue and the parent can
    wait forever. Reading as workers finish keeps the pipe drained.
    """
    results_by_user = {}

    while len(results_by_user) < len(processes):
        try:
            user_label, output = queue.get(timeout=0.5)
            results_by_user[user_label] = output
            continue
        except Empty:
            pass

        for user_label, process in processes:
            if user_label in results_by_user:
                continue
            if process.exitcode is not None:
                results_by_user[user_label] = (
                    f"ERROR: worker exited without returning output "
                    f"(exit code {process.exitcode})"
                )

    for _, process in processes:
        process.join()

    return [results_by_user[user_label] for user_label, _ in processes]

# ─────────────────────────────────────────────────────────────
# Public — called by the LangGraph tool
# ─────────────────────────────────────────────────────────────

def run_all_users(code: str, files: dict = None) -> str:
    """
    Full lifecycle for one agent request:

        0. Record activity + ensure container is running (unpause if needed)
        1. Create a unique request directory  →  /app/tmp/<uuid>/
        2. Write solution.py ONCE            →  /app/tmp/<uuid>/solution.py
        3. Copy any provided files to the request directory
        4. Spawn one Process per user        →  each runs the same file independently
        5. Wait for all processes to finish
        6. Delete the entire request directory
        7. Return combined output
    
    Args:
        code: Python code to execute
        files: Optional dict of {filename: content} to copy to sandbox
    """

    # ── 0. Track activity + wake container if paused/stopped ──
    _record_activity()
    _ensure_container_running()

    # ── 1. Unique directory per request ───────────────────────
    request_dir = f"{TMP_DIR}/{uuid.uuid4().hex}"
    filepath    = f"{request_dir}/{FILENAME}"

    # ── 2. Create the directory inside the container ──────────
    subprocess.run(
        ["docker", "exec", CONTAINER_NAME, "mkdir", "-p", request_dir],
        check=True,
        capture_output=True,
    )

    # ── 3. Write solution.py ONCE ─────────────────────────────
    wrapped_code = _wrap_code_with_plot_capture(code, request_dir)
    subprocess.run(
        ["docker", "exec", "-i", CONTAINER_NAME,
         "bash", "-c", f"cat > {filepath}"],
        input=wrapped_code,
        text=True,
        check=True,
    )

    print(f"\n  [Executor] Written  → {filepath}", flush=True)

    # ── 3b. Copy any provided files to the sandbox ─────────────
    if files:
        for filename, content in files.items():
            file_path = f"{request_dir}/{filename}"
            subprocess.run(
                ["docker", "exec", "-i", CONTAINER_NAME,
                 "bash", "-c", f"cat > {file_path}"],
                input=content,
                text=True,
                check=True,
            )
            print(f"  [Executor] Copied   → {file_path}", flush=True)

    print(f"  [Executor] Spawning {len(USERS)} processes...\n", flush=True)

    # ── 4. One Process per user — same file, separate processes ─
    queue     = Queue()
    processes = [
        (user, Process(target=_run_user, args=(user, filepath, request_dir, queue)))
        for user in USERS
    ]

    for _, p in processes:
        p.start()

    results = _collect_worker_results(processes, queue)

    # ── 5. Delete the request directory ───────────────────────
    subprocess.run(
        ["docker", "exec", CONTAINER_NAME, "rm", "-rf", request_dir],
        capture_output=True,
    )

    print(f"  [Executor] Cleaned  → {request_dir}\n", flush=True)

    # ── 6. Return results ────────────────────────────────────
    return results[0] if results else "No output"


# ── Start watchdog the moment this module is imported ─────────
start_watchdog()
