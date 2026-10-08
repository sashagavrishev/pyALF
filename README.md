# pyALF (fork)

This is a personal fork of the [pyALF](https://github.com/ALF-QMC/pyALF) package.

For documentation, installation instructions, and the full project description, please refer to the upstream repository at **https://github.com/ALF-QMC/pyALF**.

## Key fork branches

| Branch | Description |
|--------|-------------|
| `master` | Kept in sync with pyALF `master` |
| `development` | Personal development |

## Fork additions

### `ClusterSubmitter`: submitting through submitit

```python
from py_alf import ClusterSubmitter

cs = ClusterSubmitter(
    "slurm",                       # or "local" / "debug"
    slurm_mem="4G",
    partition_rules={"short": 2, "medium": 48},   # wall-time limits in hours
    slurm_mail_type="FAIL",        # any other slurm_* option passes through
)
jobs = cs.submit(sims)             # one SLURM array for several sims
```

- **Partitions:** `partition_rules` maps each partition to its wall-time limit in hours; SLURM itself rejects a request that no node can hold.
- **Wall time:** `CPU_MAX` plus 10% so ALF can finish its last bin, capped at the limit of the smallest partition that fits `CPU_MAX`.
- **Filtering:** sims whose `jobid.txt` names an active job are skipped (`skip_active=False` to trust the caller), as are those with a leftover `RUNNING` file (`stale_running="remove"` clears it).
- **Other options:** `runner=` replaces what runs on the node, `prep=False` leaves directory preparation to it, and `max_requeues=` sets submitit's requeue budget.
- **Files:** submitit's scripts, pickles and logs go to `submit_dir`, by default `.pyalf/` at the project root. A per-call `submit_dir` may contain submitit's `%A` / `%j` placeholders.

### Campaigns: chains driven to a bin target (`py_alf.campaign`)

```python
from py_alf.campaign import Campaign, Chain, SegmentPolicy, ledger_path

chains = [Chain.from_sim(sim, target_bins=400, array_key="L8") for sim in sims]
campaign = Campaign(
    name="default",
    chains=chains,
    target_bins=400,
    submitter=cs,
    ledger_path=ledger_path(data_dir, "default"),
    jobs_dir=data_dir / "campaigns" / "default" / "jobs",
    policy=SegmentPolicy(max_partition="medium"),
    cost_model=lambda sim_dict: 0.05,   # hours per bin before a chain measures itself
)
campaign.launch()                       # one array per array_key
campaign.status()                       # per-chain bins and verdict
campaign.reconcile()                    # resubmit what stalled
campaign.log_path(chain_id)             # submitit log of a chain's latest segment
campaign.cancel()                       # scancel every live array
campaign.prune()                        # drop superseded job folders and pickles
Campaign.from_ledger(ledger_path(data_dir, "default"), cs).status()   # from a fresh shell
```

Three layers carry each chain to its target:
1. `CPU_MAX`, sized per segment by `SegmentPolicy`, stops ALF at a bin boundary inside the partition limit.
2. submitit requeues a task the wall clock or a preemption cut short.
3. `reconcile` resubmits what neither delivered.

Under submitit a wall-clock stop is recorded as `FAILED` or `CANCELLED`, never `TIMEOUT`, so progress is judged by bins on disk. Chains whose array logs show a timeout are reported as `TIMEOUT`.

```
<data_dir>/campaigns/<name>.json                    the ledger: chains, seeds, segments
<data_dir>/campaigns/<name>/jobs/<array_key>/<array id>/   submitit files, one folder per array
<sim_dir>/segments/<job id>-<time>.json             what each segment did, written on the node
```

### `Simulation` and `ALF_source` additions

- `Simulation(..., mc_seed=)` writes the seed as the first line of `seeds`, which makes separate single-core jobs independent Markov chains.
- `Simulation(..., env={...})` adds environment variables to the ALF process (e.g. `ALF_DELAY_K`), locally and on the node.
- `Simulation.bin_count(counting_obs="Ener_scal")` counts the bins in its `data.h5`.
- `ALF_source.commit()` reads the commit ALF was last built at from `Prog/git.h`.

### Module map

| Module | Contents |
|---|---|
| `cluster_submission` | `ClusterSubmitter` |
| `campaign` | `Campaign`, `Chain`, `ChainStatus`, `Ledger`, `SegmentPolicy`, `chain_id`, `ledger_path` |
| `slurm` | `job_states`, `queued_arrays`, `is_timeout`, `job_log`, `cancel`, `ACTIVE_STATES`, `TERMINAL_STATES` |
| `bins` | `read_bin_count`, `read_bin_counts` |
| `execute` | `exec_alf_binary`, the one place ALF is started |
