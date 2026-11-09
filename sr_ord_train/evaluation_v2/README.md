# Evaluation pair launcher v2

This version corrects the Slurm step request used by `evaluation_v1`. It invokes
unchanged `adapter.evaluation worker` with the existing registered endpoint
identities and `worker_shards=1`. FP32, seeds, sampling, request budgets, and
checkpoint loading are unchanged. Keep v1 and all existing submission evidence.

A typed job allocation combined with an untyped step request fails with
`Invalid GRES specification (with and without type identification)`. Preserve
any failed submission and establish whether a worker or model call occurred
before considering recovery. This launcher neither resubmits nor erases evidence.

The job requests two typed `pro6000` GPUs. Each v2 step explicitly requests
`--gpus-per-task=pro6000:1`, with `--exclusive --exact`, one task, four CPUs, and
33792 MiB. The initial job guard still requires two distinct visible GPUs,
eight CPUs, and 67584 MiB. Both workers are awaited even when one fails.

Slurm's [GRES documentation](https://slurm.schedmd.com/gres.html) requires GPU type
consistency between allocations and steps. Its [srun documentation](https://slurm.schedmd.com/srun.html)
describes inherited environment variables that act as resource or binding options.
The launcher removes these inputs only for each `srun` invocation:

- GPU counts/GRES: `SLURM_GPUS`, `SLURM_GPUS_PER_NODE`, `SLURM_GPUS_PER_TASK`,
  `SLURM_GRES`, `SLURM_STEP_GRES`, `SLURM_GRES_FLAGS`.
- Task/binding: `SLURM_TRES_PER_TASK`, `SLURM_GPU_BIND`, `SLURM_TRES_BIND`,
  `SLURM_NTASKS_PER_GPU`.
- Conflicting per-resource CPU/memory options: `SLURM_CPUS_PER_GPU`,
  `SLURM_MEM_PER_GPU`, `SLURM_MEM_PER_CPU`.

`CUDA_VISIBLE_DEVICES`, `SLURM_JOB_ID`, `SLURM_JOB_GPUS`, and
`SLURM_GPUS_ON_NODE` retain the allocation identity. Slurm assigns each exclusive
step its visible GPU; the unchanged evaluation backend checks one visible GPU.
Actual worker UUID separation still requires runtime acceptance.

Run local synthetic checks with:

```sh
python3 -B -m unittest discover -s adapter_tests -p 'test_eval_pair_launcher*.py' -v
bash -n evaluation_v2/run_endpoint_pair.sh
```

Tests mock typed job inputs, reject untyped step requests before worker execution,
check option cleanup and retained allocation identity, and cover fresh-output
protection plus failure/peer waiting. They do not establish real Slurm admission,
model feasibility, or evaluation completion. Deployment and each submission must
bind the reviewed v2 script hash and pass the current submission audit separately.
