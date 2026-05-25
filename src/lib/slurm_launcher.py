import shlex
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from mirror.slurm_util import SlurmConfig
from mirror.util import is_login_node


def submit_slurm_job(
    slurm: SlurmConfig,
    activate_cmd: str = "",
    python_cmd: str = "uv run python"
) -> None:
    """Submit a slurm job and exit, if `slurm.job_type` is 'compute' on a login node.

    `activate_cmd` is the shell command used to
    enter the project's environment inside the submitted job — e.g.
    `"source .venv/bin/activate"` for uv, `"mamba activate ./.env"` for mamba,
    or `""` to skip activation (e.g. when `python_cmd` already wraps with `uv run`).
    Returns normally if no job was submitted; calls `sys.exit` after a
    successful submission so the caller doesn't have to bail out.
    """
    if not is_login_node() or slurm.job_type != "compute":
        return

    env = Environment(
        loader=FileSystemLoader(Path(__file__).parent / "templates"),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    python_args = sys.argv[1:]
    script = env.get_template("slurm.jinja").render(
        **asdict(slurm),
        chdir=str(Path.cwd()),
        activate_cmd=activate_cmd,
        run_cmd=f"srun {python_cmd} {sys.argv[0]} {shlex.join(python_args)}",
    )

    res = subprocess.run(["sbatch"], input=script, text=True, capture_output=True)
    if res.returncode != 0:
        raise RuntimeError(
            f"sbatch failed (exit {res.returncode}):\n{res.stderr}\n\nGenerated script:\n{script}"
        )
    job_id = res.stdout.strip().split()[-1]
    print(f"Submitted batch job {job_id}")
    sys.exit(0)
