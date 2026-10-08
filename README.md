# pyALF (fork)

This is a personal fork of the [pyALF](https://github.com/ALF-QMC/pyALF) package.

For documentation, installation instructions, and the full project description, please refer to the upstream repository at **https://github.com/ALF-QMC/pyALF**.

## Key fork branches

| Branch | Description |
|--------|-------------|
| `master` | Kept in sync with pyALF `master` |
| `development` | Personal development |

## Fork additions

### Detect partition rules

Functionality exists to be able automatically detect the required partition rules on a SLURM cluster, using `detect_partition_rules`.

```python
from py_alf import detect_partition_rules, ClusterSubmitter

rules = detect_partition_rules(exclude=["gpu", "debug"])
cs = ClusterSubmitter("slurm", slurm_mem="8G", partition_rules=rules)
```
