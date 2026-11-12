import json
import subprocess


def run_worker(
    python_executable,
    worker_script,
    request,
    cwd=None,
    timeout=600,
):
    command = [str(python_executable), str(worker_script)]

    process = subprocess.run(
        command,
        input=json.dumps(request),
        capture_output=True,
        text=True,
        cwd=str(cwd) if cwd else None,
        timeout=timeout,
        check=False,
    )

    if process.returncode != 0:
        raise RuntimeError(
            f"Worker failed: {worker_script}\n"
            f"STDOUT:\n{process.stdout[-3000:]}\n"
            f"STDERR:\n{process.stderr[-3000:]}"
        )

    output_lines = [
        line.strip()
        for line in process.stdout.splitlines()
        if line.strip()
    ]

    if not output_lines:
        raise RuntimeError("Worker returned no output.")

    try:
        return json.loads(output_lines[-1])
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Worker returned invalid JSON:\n{process.stdout[-3000:]}"
        ) from exc
