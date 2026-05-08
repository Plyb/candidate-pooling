import sys

from mirror.slurm_launcher import submit_slurm_job
from mirror.slurm_util import SlurmConfig
from mirror.util import is_login_node

MODEL_ID = "meta-llama/Llama-3.2-1B"


def _prefetch() -> None:
    from byutils import prefetch_dataset, prefetch_model

    prefetch_model(MODEL_ID)
    prefetch_dataset("cais/mmlu", "auxiliary_train")


def _run() -> None:
    from candidate_pooling.pipeline import run_pipeline

    run_pipeline()


def main() -> None:
    if "--prefetch" in sys.argv and is_login_node():
        _prefetch()

    slurm = SlurmConfig(
        job_type="compute",
        time="06:00:00",
        gpus_per_node="a100:1",
        mem_per_cpu="32G",
        qos="dw87",
    )
    submit_slurm_job(slurm, sys.argv[1:])
    _run()  # only reached on compute node (login node exits via sys.exit)


if __name__ == "__main__":
    main()
