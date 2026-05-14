import os
import time
import uuid
import threading
import subprocess
from multiprocessing import Process, Queue

# ── Config ────────────────────────────────────────────────────
CONTAINER_NAME    = "sandbox-merged"
DOCKER_IMAGE      = "sandbox-merged"
PYTHON_BIN        = "/.venv/bin/python3"
TMP_DIR           = "/app/tmp"
FILENAME          = "solution.py"
USERS             = ["A", "B"]

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
        subprocess.run([
            "docker", "run",
            "-d",                       # detached
            "--name", CONTAINER_NAME,
            DOCKER_IMAGE,
            "sleep", "infinity",        # keep it alive indefinitely
        ], check=True, capture_output=True)
        time.sleep(2)

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
        _start_container()


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


# ─────────────────────────────────────────────────────────────
# Private — runs inside each spawned Process
# ─────────────────────────────────────────────────────────────

def _run_user(user_label: str, filepath: str, queue: Queue):
    """
    Each Process calls this independently.
    All processes point to the same solution.py inside the request directory.
    Each gets its own OS PID and memory space — truly parallel.
    """
    result = subprocess.run(
        ["docker", "exec", CONTAINER_NAME, PYTHON_BIN, filepath],
        capture_output=True,
        text=True,
    )

    if result.returncode == 0:
        output = result.stdout.strip() or "✓ (no output)"
    else:
        # Show both stdout and stderr so errors are never silently empty
        stdout = result.stdout.strip()
        stderr = result.stderr.strip()
        output = f"ERROR: {stderr or stdout or 'unknown error (no output captured)'}"

    print(f"  [User-{user_label}] PID={os.getpid()} → {output}", flush=True)
    queue.put(f"User-{user_label} (PID={os.getpid()}): {output}")

# ─────────────────────────────────────────────────────────────
# Public — called by the LangGraph tool
# ─────────────────────────────────────────────────────────────

def run_all_users(code: str) -> str:
    """
    Full lifecycle for one agent request:

        0. Record activity + ensure container is running (unpause if needed)
        1. Create a unique request directory  →  /app/tmp/<uuid>/
        2. Write solution.py ONCE            →  /app/tmp/<uuid>/solution.py
        3. Spawn one Process per user        →  each runs the same file independently
        4. Wait for all processes to finish
        5. Delete the entire request directory
        6. Return combined output
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
    subprocess.run(
        ["docker", "exec", "-i", CONTAINER_NAME,
         "bash", "-c", f"cat > {filepath}"],
        input=code,
        text=True,
        check=True,
    )

    print(f"\n  [Executor] Written  → {filepath}", flush=True)
    print(f"  [Executor] Spawning {len(USERS)} processes...\n", flush=True)

    # ── 4. One Process per user — same file, separate processes ─
    queue     = Queue()
    processes = [
        Process(target=_run_user, args=(user, filepath, queue))
        for user in USERS
    ]

    for p in processes:
        p.start()

    for p in processes:
        p.join()

    # ── 5. Delete the request directory ───────────────────────
    subprocess.run(
        ["docker", "exec", CONTAINER_NAME, "rm", "-rf", request_dir],
        capture_output=True,
    )

    print(f"  [Executor] Cleaned  → {request_dir}\n", flush=True)

    # ── 6. Collect and return results ─────────────────────────
    results = [queue.get() for _ in processes]
    return "\n".join(sorted(results))


# ── Start watchdog the moment this module is imported ─────────
start_watchdog()