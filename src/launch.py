import sys


from lib.slurm_launcher import submit_slurm_job
from mirror.slurm_util import SlurmConfig
from mirror.util import is_login_node



def _prefetch() -> None:
    from byutils import prefetch_dataset, prefetch_model
    from candidate_pooling.pipeline import MODEL_ID
    from sae_lens import SAE
    from candidate_pooling.pipeline import MINING_STRATEGY, SaeStrategy
    if isinstance(MINING_STRATEGY, SaeStrategy):
        SAE.from_pretrained(MINING_STRATEGY.release, MINING_STRATEGY.sae_id) # TODO: put this in the right cache location


    prefetch_model(MODEL_ID)
    prefetch_dataset("allenai/ai2_arc", "ARC-Easy")


def _run() -> None:
    from candidate_pooling.pipeline import run_pipeline
    
    run_pipeline(850, 200)


def main() -> None:
    if "--prefetch" in sys.argv and is_login_node():
        _prefetch()

    slurm = SlurmConfig(
        job_type="compute",
        time="01:00:00",
        gpus_per_node="h100:1",
        mem_per_cpu="128G",
        qos="cs",
        nodes=1,
        ntasks_per_node=1,
    )
    submit_slurm_job(slurm)
    _run()  # only reached on compute node (login node exits via sys.exit)


if __name__ == "__main__":
    main()
