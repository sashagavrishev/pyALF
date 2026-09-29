#!/usr/bin/env python3
"""
Visual demo for the SimulationMonitor TUI — no ALF installation required.

Run from the repo root:

    python demos/demo_monitor_tui.py

Demo modes
----------
  0  fresh sims — launches the monitor directly with a pre-built mix of
                  SLURM statuses (RUNNING, PENDING, COMPLETED, FAILED,
                  CANCELLED, INACTIVE)
  1  from session — writes a session manifest to disk and loads it via
                    SimulationMonitor.from_session(), exercising the same
                    code path used after a real SubmissionReview run
  2  parallel params — a single PARALLEL_PARAMS job (one SLURM id) is expanded
                       into one row per Temp_i parameter set, each with
                       its own bin progress.  Tagged PP[i] and flagged in the
                       title bar so it is not mistaken for a SLURM array.  One
                       job is RUNNING (live per-config bars) and one COMPLETED
                       (press 'i' to see each rank's distinct seed).

All SLURM queries, bin counts, and cluster operations are mocked so no
cluster connection is needed.

Controls inside the TUI
-----------------------
  q     quit
  i     view ALF info file  (only active on COMPLETED rows)
  l     view log (shows fake ALF output for jobs that have a log file)
  c     cancel individual job    (mocked — disabled for terminal states)
  a     cancel SLURM array       (mocked — disabled for terminal states)
  r     resubmit selected sim    (mocked)
  f     manual refresh
"""

from __future__ import annotations

import atexit
import json
import shutil
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from py_alf.cluster_submission import ClusterSubmitter
from py_alf.monitor import SimulationMonitor

# ---------------------------------------------------------------------------
# Temporary workspace — wiped on exit
# ---------------------------------------------------------------------------

_TMPDIR = Path(tempfile.mkdtemp(prefix="alf_demo_monitor_"))
atexit.register(shutil.rmtree, _TMPDIR, ignore_errors=True)

_SUBMIT_DIR = _TMPDIR / ".alfmonitor"
_SUBMIT_DIR.mkdir()

# ---------------------------------------------------------------------------
# Mock SLURM data
# ---------------------------------------------------------------------------

ARRAY_ID = "44321"

# job_id → (status, runtime, n_bins)
_SIM_TABLE: list[tuple[str, str, str | None, int]] = [
    # ham_name,       job_id             status       runtime       n_bins
    ("Hubbard_PV", f"{ARRAY_ID}_0", "RUNNING", "2:15:30", 18),
    ("Hubbard_PV", f"{ARRAY_ID}_1", "RUNNING", "2:14:02", 15),
    ("Hubbard_PV", f"{ARRAY_ID}_2", "PENDING", None, 0),
    ("Hubbard_PV", f"{ARRAY_ID}_3", "COMPLETED", "1:58:44", 40),
    ("Hubbard_PV", f"{ARRAY_ID}_4", "COMPLETED", "2:02:17", 40),
    ("Hubbard_PV", f"{ARRAY_ID}_5", "FAILED", "0:03:11", 3),
    ("tV_Model", f"{ARRAY_ID}_6", "CANCELLED", "0:00:05", 0),
    ("tV_Model", None, "INACTIVE", None, 0),  # no job ID
]

_BETA_VALUES = [1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 0.5, 1.0]
_L_VALUES = [4, 4, 4, 4, 4, 4, 8, 8]

CPU_PARTITIONS: dict[str, float] = {
    "short": 2,
    "medium": 48,
    "long": 336,
    "extra_long": 672,
}


# ---------------------------------------------------------------------------
# Build temp filesystem (sim dirs + jobid.txt + fake logs)
# ---------------------------------------------------------------------------


def _setup_workspace() -> tuple[list, dict, dict]:
    """
    Create sim dirs, write jobid.txt where applicable, write fake log files.

    Returns
    -------
    sims      : list of lightweight sim-like objects
    statuses  : dict for _get_slurm_status_bulk mock
    bin_map   : sim_dir → n_bins for _bin_count mock
    """
    from types import SimpleNamespace

    sims = []
    statuses: dict[str, dict] = {}
    bin_map: dict[str, int] = {}

    for i, (ham, jid, status, runtime, n_bins) in enumerate(_SIM_TABLE):
        beta = _BETA_VALUES[i]
        L = _L_VALUES[i]
        sim_dir = _TMPDIR / "ALF_data" / f"{ham}_Beta={beta}_L={L}"
        sim_dir.mkdir(parents=True, exist_ok=True)

        if jid is not None:
            (sim_dir / "jobid.txt").write_text(jid)
            nodelist = f"compute{i:02d}" if status == "RUNNING" else None
            statuses[jid] = {"status": status, "runtime": runtime, "nodelist": nodelist}
            if status in ("RUNNING", "COMPLETED"):
                log = _SUBMIT_DIR / f"{jid}_0_log.out"
                log.write_text(_fake_log(jid, ham, beta, L, status))
            if status == "COMPLETED":
                (sim_dir / "info").write_text(_fake_info(ham, beta, L))
        else:
            # INACTIVE: leave a RUNNING sentinel file to trigger CRASHED display
            # if you want to test that path, uncomment the next line:
            # (sim_dir / "RUNNING").touch()
            pass

        sim = SimpleNamespace(
            ham_name=ham,
            sim_dir=str(sim_dir),
            sim_dict={"Beta": beta, "L1": L, "L2": L, "CPU_MAX": 24},
            n_omp=4,
            n_mpi=1,
            mpi=False,
            mpiexec="mpiexec",
            mpiexec_args=[],
            config="GNU HDF5 NOMPI",
        )
        sims.append(sim)
        bin_map[str(sim_dir)] = n_bins

    return sims, statuses, bin_map


def _fake_log(jid: str, ham: str, beta: float, L: int, status: str) -> str:
    lines = [
        f"[ALF] job {jid}  ham={ham}  Beta={beta}  L={L}",
        "[ALF] Initialising Hamiltonian...",
        "[ALF] Warmup phase  — 1000 sweeps",
        "[ALF] Production phase — 5000 sweeps",
    ]
    if status == "COMPLETED":
        lines += [
            "[ALF] Writing observables to data.h5",
            "[ALF] Done. All bins collected.",
        ]
    else:
        lines += ["[ALF] Still running..."]
    return "\n".join(lines) + "\n"


def _fake_info(ham: str, beta: float, L: int) -> str:
    return (
        f"Hamiltonian : {ham}\n"
        f"Beta        : {beta}\n"
        f"L1 / L2     : {L} / {L}\n"
        f"OMP threads : 4\n"
        f"Bins done   : 40\n"
        f"Status      : COMPLETED\n"
        f"Observables : Kin_Energy, Pot_Energy, Density, SpinZ\n"
    )


# ---------------------------------------------------------------------------
# PARALLEL_PARAMS workspace (mode 2): one job, many Temp_i parameter sets
# ---------------------------------------------------------------------------

# ham,      U,    job_id,  status,      runtime,   NBin, bins per Temp_i
_PP_JOBS: list[tuple[str, float, str, str, str | None, int, list[int]]] = [
    (
        "Hubbard",
        0.5,
        "55123",
        "RUNNING",
        "1:12:40",
        40,
        [22, 18, 25, 30, 15, 28],
    ),
    (
        "Hubbard",
        1.0,
        "55124",
        "COMPLETED",
        "1:58:03",
        40,
        [40, 40, 40, 40, 40, 40],
    ),
]


def _setup_parallel_params_workspace() -> tuple[list, dict, dict]:
    """One SLURM job per entry, each prepared with len(bins) Temp_i/ dirs.

    Mirrors a real PARALLEL_PARAMS submission: a single job id, one MPI rank per
    parameter set writing into Temp_i/.  The monitor expands each job into
    one row per Temp_i.
    """
    from types import SimpleNamespace

    sims: list = []
    statuses: dict[str, dict] = {}
    bin_map: dict[str, int] = {}

    for ham, U, jid, status, runtime, nbin, bins in _PP_JOBS:
        n_real = len(bins)
        sim_dir = _TMPDIR / "ALF_data" / f"{ham}_U={U}"
        sim_dir.mkdir(parents=True, exist_ok=True)
        (sim_dir / "jobid.txt").write_text(jid)

        nodelist = "compute[01-02]" if status == "RUNNING" else None
        statuses[jid] = {"status": status, "runtime": runtime, "nodelist": nodelist}
        if status in ("RUNNING", "COMPLETED"):
            log = _SUBMIT_DIR / f"{jid}_0_log.out"
            log.write_text(_fake_pp_log(jid, ham, U, status, n_real))

        for i, n_bins in enumerate(bins):
            temp = sim_dir / f"Temp_{i}"
            temp.mkdir(parents=True, exist_ok=True)
            bin_map[str(temp)] = n_bins
            # Per-rank info shows the distinct seed (base + igroup); only
            # meaningful once COMPLETED, when 'i' is enabled in the TUI.
            if status == "COMPLETED":
                (temp / "info").write_text(_fake_pp_info(ham, U, seed=i))

        sims.append(
            SimpleNamespace(
                ham_name=ham,
                sim_dir=str(sim_dir),
                sim_dict=[
                    {
                        "Beta": 5.0,
                        "L1": 8,
                        "L2": 8,
                        "NBin": nbin,
                        "Ham_U": U,
                        "mpi_per_parameter_set": 1,
                    }
                ]
                * n_real,
                n_omp=1,
                n_mpi=n_real,
                mpi=True,
                mpiexec="mpiexec",
                mpiexec_args=[],
                config="GNU PARALLEL_PARAMS HDF5 NO-INTERACTIVE",
            )
        )

    return sims, statuses, bin_map


def _fake_pp_log(jid: str, ham: str, U: float, status: str, n_real: int) -> str:
    lines = [
        f"[ALF] job {jid}  PARALLEL_PARAMS  ham={ham}  U={U}",
        f"[ALF] {n_real} parameter sets, one MPI rank each "
        f"→ Temp_0 … Temp_{n_real - 1}",
    ]
    lines += [
        "[ALF] All parameter sets done." if status == "COMPLETED" else "[ALF] Running…"
    ]
    return "\n".join(lines) + "\n"


def _fake_pp_info(ham: str, U: float, seed: int) -> str:
    return (
        f"Model is                   : {ham}\n"
        f"Ham_U                      : {U}\n"
        f"Seed (base+igroup)         : {seed}\n"
        f"Beta                       : 5.0\n"
        f"L1 / L2                    : 8 / 8\n"
        f"Status                     : COMPLETED\n"
    )


# ---------------------------------------------------------------------------
# Mock patches
# ---------------------------------------------------------------------------


def _mock_bin_count_factory(bin_map: dict[str, int]):
    def _mock(sim, data_dir=None, **_kw):
        # The monitor passes the per-row directory as data_dir (the parent
        # sim_dir for a normal row, or Temp_i/ for a PARALLEL_PARAMS row), so
        # keying on it works for both ordinary and expanded jobs.
        key = data_dir if data_dir is not None else sim.sim_dir
        return bin_map.get(str(key), 0)

    return _mock


def _mock_resubmit(self: ClusterSubmitter, sims, **kwargs) -> list:
    sims = list(sims) if not hasattr(sims, "sim_dir") else [sims]
    print(f"\n[MOCK] resubmit: {[str(s.sim_dir) for s in sims]}")
    jobs = []
    for _s in sims:
        j = MagicMock()
        j.job_id = "99999_resubmit"
        jobs.append(j)
    return jobs


# ---------------------------------------------------------------------------
# Session manifest for mode 1
# ---------------------------------------------------------------------------


def _write_demo_session(sims: list, statuses: dict) -> Path:
    """Write a session JSON that from_session() can load."""
    entries = []
    for s in sims:
        jid_file = Path(s.sim_dir) / "jobid.txt"
        entries.append(
            {
                "sim_dir": s.sim_dir,
                "job_id": jid_file.read_text().strip() if jid_file.exists() else None,
                "ham_name": s.ham_name,
                "n_omp": s.n_omp,
                "n_mpi": s.n_mpi,
                "mpi": s.mpi,
                "mpiexec": s.mpiexec,
                "mpiexec_args": s.mpiexec_args,
                "sim_dict": s.sim_dict,
            }
        )

    manifest = {
        "version": 1,
        "submitted_at": "2026-05-13T10:00:00",
        "cluster_submitter": {
            "executor": "slurm",
            "submit_dir": str(_SUBMIT_DIR),
            "slurm_mem": "8G",
            "partition_rules": CPU_PARTITIONS,
            "job_name": "monitor_demo",
            "mail_type": "END",
            "wckey": None,
            "stderr_to_stdout": False,
            "slurm_kwargs": {},
        },
        "entries": entries,
    }
    path = _SUBMIT_DIR / "session_demo.json"
    path.write_text(json.dumps(manifest, indent=2))
    return path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

MODES = {
    "fresh sims (direct SimulationMonitor)": "fresh",
    "from session manifest (SimulationMonitor.from_session)": "session",
    "PARALLEL_PARAMS expansion (one row per Temp_i)": "parallel_params",
}


def main() -> None:
    names = list(MODES)
    print("Demo modes:\n")
    for i, name in enumerate(names):
        print(f"  {i}  {name}")
    raw = input(f"\nPick mode [0–{len(names) - 1}, default 0]: ").strip()
    idx = int(raw) if raw.isdigit() and int(raw) < len(names) else 0
    mode = list(MODES.values())[idx]

    if mode == "parallel_params":
        sims, statuses, bin_map = _setup_parallel_params_workspace()
    else:
        sims, statuses, bin_map = _setup_workspace()

    print(f"\nTemp workspace: {_TMPDIR}\n")

    cs = ClusterSubmitter(
        "slurm",
        submit_dir=str(_SUBMIT_DIR),
        slurm_mem="8G",
        partition_rules=CPU_PARTITIONS,
        job_name="monitor_demo",
    )

    with (
        patch("py_alf.monitor._get_slurm_status_bulk", return_value=statuses),
        patch(
            "py_alf.monitor._bin_count", side_effect=_mock_bin_count_factory(bin_map)
        ),
        patch("py_alf.monitor.cancel_cluster_job", return_value=True),
        patch.object(ClusterSubmitter, "submit", _mock_resubmit),
    ):
        if mode == "session":
            session_path = _write_demo_session(sims, statuses)
            print(f"Session manifest: {session_path}\n")
            monitor = SimulationMonitor.from_session(
                session_path,
                cluster_submitter=cs,
                param_keys=["Beta", "L1"],
            )
        else:
            # "fresh" and "parallel_params" both launch the monitor directly;
            # the PARALLEL_PARAMS expansion is driven entirely by each sim's
            # config tag and its Temp_i/ dirs on disk.
            monitor = SimulationMonitor(
                sims,
                cluster_submitter=cs,
                submit_dir=str(_SUBMIT_DIR),
                param_keys=["Beta", "L1"],
            )

        monitor.run()


if __name__ == "__main__":
    main()
