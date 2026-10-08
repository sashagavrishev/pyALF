"""Tests for the campaign layer in py_alf.campaign.

These cover the decisions a campaign makes without running ALF: how long a
segment may ask for, what a chain is called, what the ledger remembers, and how
a chain's progress is judged. Anything needing a real ALF binary lives in the
consuming project's integration tests.
"""

import json
import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import submitit

from py_alf.bins import read_bin_counts
from py_alf.campaign import Ledger, SegmentPolicy, chain_id
from py_alf.campaign.campaign import Campaign, ChainStatus
from py_alf.campaign.chain import Chain
from py_alf.campaign.worker import _claim_running, _clear_own_running, run_segment
from py_alf.simulation import Simulation

# A three-tier cluster of the shape these policies exist to cope with.
RULES = {
    "short": {"max_hours": 2},
    "medium": {"max_hours": 48},
    "long": {"max_hours": 336},
}


# --- SegmentPolicy ----------------------------------------------------------


def test_cpu_max_never_exceeds_the_capped_partition():
    """A budget request must stay inside `medium`, however much work is left."""
    policy = SegmentPolicy()
    cap = policy.max_hours(RULES)
    assert cap == pytest.approx(48 * 0.95)
    for remaining, hours_per_bin in [(1, 0.5), (100, 0.5), (100_000, 10.0)]:
        assert policy.cpu_max(remaining, hours_per_bin, RULES) <= cap


def test_cpu_max_shrinks_to_the_work_that_is_left():
    """A nearly-finished chain asks for little, so it can land in a fast queue."""
    policy = SegmentPolicy(min_hours=0.0)
    small = policy.cpu_max(1, 0.5, RULES)
    large = policy.cpu_max(100, 0.5, RULES)
    assert small < large
    assert small == pytest.approx(1 * 0.5 * policy.safety)


def test_cpu_max_respects_the_floor():
    """Below the floor the queue wait dwarfs the run, so do not ask for less."""
    policy = SegmentPolicy(min_hours=0.25)
    assert policy.cpu_max(1, 1e-9, RULES) == pytest.approx(0.25)


def test_max_hours_unbounded_without_partition_rules():
    """Local and debug executors define no partitions, so nothing caps them."""
    assert SegmentPolicy().max_hours({}) == float("inf")


def test_segments_needed_grows_with_the_work():
    policy = SegmentPolicy()
    assert policy.segments_needed(1, 0.01, RULES) == 1
    assert policy.segments_needed(100, 2.0, RULES) > 1
    # However bad the estimate, never queue more attempts than allowed.
    assert policy.segments_needed(10**9, 10.0, RULES) <= policy.max_segments


def test_unknown_max_partition_is_rejected():
    with pytest.raises(SystemExit):
        SegmentPolicy(max_partition="nonexistent").max_hours(RULES)


# --- chain_id ---------------------------------------------------------------


def _sim(sim_dir, ham_name="Hubbard"):
    """Stand-in for a Simulation: chain_id reads only these two attributes.

    A real Simulation would need an ALF source tree to validate parameters
    against, which these unit tests deliberately do not require.
    """
    stub = MagicMock()
    stub.__class__ = Simulation
    stub.ham_name = ham_name
    stub.sim_dir = str(sim_dir)
    return stub


def test_chain_from_sim_derives_its_id_and_seed(tmp_path):
    sim = _sim(tmp_path / "Hubbard_L1=4")
    sim.mc_seed = 7
    chain = Chain.from_sim(sim, 100, point={"N": 4}, array_key="k")

    assert chain.chain_id == chain_id(sim, 7)
    assert chain.mc_seed == 7
    assert chain.to_record()["mc_seed"] == 7
    assert (chain.target_bins, chain.point, chain.array_key) == (100, {"N": 4}, "k")


def test_chain_id_is_deterministic_and_unique(tmp_path):
    a = _sim(tmp_path / "Hubbard_L1=4")
    b = _sim(tmp_path / "Hubbard_L1=6")
    assert chain_id(a, 1) == chain_id(a, 1)
    assert chain_id(a, 1) != chain_id(b, 1), "parameters must distinguish chains"
    assert chain_id(a, 1) != chain_id(a, 2), "mc_seed must distinguish chains"
    assert chain_id(a, 1) != chain_id(_sim(tmp_path / "Hubbard_L1=4", "Other"), 1)


def test_chain_id_survives_relocating_the_data(tmp_path):
    """The id names the chain, not where its output happens to live.

    It hashes the directory's *name* -- already a canonical rendering of every
    Hamiltonian parameter -- so moving the data root renumbers nothing.
    """
    here = _sim(tmp_path / "here" / "Hubbard_L1=4")
    there = _sim(tmp_path / "there" / "Hubbard_L1=4")
    assert chain_id(here, 1) == chain_id(there, 1)


# --- Ledger -----------------------------------------------------------------


def _ledger(tmp_path, **kwargs):
    return Ledger.new(
        tmp_path / "c.json",
        name="c",
        target_bins=100,
        policy={},
        **kwargs,
    )


def test_ledger_round_trips(tmp_path):
    led = _ledger(tmp_path)
    led.data["chains"]["abc"] = {"sim_dir": "/d", "point": {"disorder_seed": 5}}
    led.save()
    assert Ledger.load(tmp_path / "c.json").chains["abc"]["point"]["disorder_seed"] == 5


def test_ledger_save_is_atomic(tmp_path):
    """A crashed driver must never leave a half-written index behind."""
    led = _ledger(tmp_path)
    led.save()
    assert not list(tmp_path.glob("*.tmp"))
    json.loads((tmp_path / "c.json").read_text())


def test_loading_a_missing_ledger_is_a_clean_error(tmp_path):
    with pytest.raises(SystemExit):
        Ledger.load(tmp_path / "absent.json")


def test_absorb_segment_records_merges_worker_output(tmp_path):
    """Only the node knows what a segment actually did; fold that back in."""
    sim_dir = tmp_path / "sim"
    (sim_dir / "segments").mkdir(parents=True)
    (sim_dir / "segments" / "000-7_0-x.json").write_text(
        json.dumps({"job_id": "7_0", "bins_before": 0, "bins_after": 40})
    )
    led = _ledger(tmp_path)
    led.data["chains"]["a"] = {
        "sim_dir": str(sim_dir),
        "point": {},
        "segments": [{"job_id": "7_0"}],
    }
    assert led.absorb_segment_records() == 1
    assert led.chains["a"]["segments"][0]["bins_after"] == 40


def test_absorb_segment_records_ignores_unreadable_files(tmp_path):
    """A record truncated by a hard kill must not break the whole scan."""
    sim_dir = tmp_path / "sim"
    (sim_dir / "segments").mkdir(parents=True)
    (sim_dir / "segments" / "000-7_0-x.json").write_text("{ truncated")
    led = _ledger(tmp_path)
    led.data["chains"]["a"] = {"sim_dir": str(sim_dir), "point": {}, "segments": []}
    led.absorb_segment_records()  # must not raise


# --- worker: checkpoint and the RUNNING mutex -------------------------------


def test_run_segment_is_checkpointable(tmp_path):
    """Without this attribute submitit fails a timed-out job instead of requeuing."""
    checkpoint = getattr(run_segment, "checkpoint", None) or getattr(
        run_segment, "__submitit_checkpoint__", None
    )
    assert checkpoint is not None
    resumed = checkpoint(_sim(tmp_path))
    assert isinstance(resumed, submitit.helpers.DelayedSubmission)
    assert resumed.function is run_segment


def test_clear_own_running_removes_only_this_job_s_file(tmp_path):
    """A requeued attempt reuses the job id, so that file is provably ours."""
    _claim_running(tmp_path, "42_1")
    (tmp_path / "RUNNING").write_text("ALF is running")
    assert _clear_own_running(tmp_path, "42_1") is True
    assert not (tmp_path / "RUNNING").exists()


def test_clear_own_running_leaves_a_foreign_file_alone(tmp_path):
    """RUNNING is a mutex: another job's ALF must be allowed to abort us."""
    _claim_running(tmp_path, "999_0")
    (tmp_path / "RUNNING").write_text("ALF is running")
    assert _clear_own_running(tmp_path, "42_1") is False
    assert (tmp_path / "RUNNING").exists()


def test_clear_own_running_leaves_an_unowned_file_alone(tmp_path):
    """No owner marker means we cannot prove it is ours, so do not touch it."""
    (tmp_path / "RUNNING").write_text("ALF is running")
    assert _clear_own_running(tmp_path, "42_1") is False
    assert (tmp_path / "RUNNING").exists()


def test_clear_own_running_is_a_noop_without_a_running_file(tmp_path):
    assert _clear_own_running(tmp_path, "42_1") is False


# --- Campaign.status: progress is judged by bins, not by SLURM --------------


def _campaign(tmp_path, chains=()):
    # status() reads submit_dir to look for submitit's timeout markers; the log
    # is absent here, which is the "cannot tell yet" path.
    submitter = MagicMock()
    submitter.submit_dir = tmp_path / "submit"
    submitter.partition_rules = RULES
    return Campaign(
        name="c",
        chains=list(chains),
        target_bins=100,
        submitter=submitter,
        ledger_path=tmp_path / "c.json",
    )


def _status_with(tmp_path, bins, segments, slurm_state=None):
    """Run Campaign.status against a ledger with one chain in a chosen state."""
    led = _ledger(tmp_path)
    led.data["chains"]["a"] = {
        "sim_dir": str(tmp_path / "sim"),
        "point": {},
        "segments": segments,
    }
    led.save()
    states = {}
    if slurm_state is not None and segments:
        states = {segments[-1]["job_id"]: {"status": slurm_state}}

    camp = _campaign(tmp_path)
    with (
        patch("py_alf.campaign.campaign.job_states", return_value=states),
        patch(
            "py_alf.campaign.campaign.read_bin_counts",
            side_effect=lambda paths, *a, **k: [bins] * len(paths),
        ),
    ):
        return camp.status(Ledger.load(tmp_path / "c.json"))[0]


def test_status_done_when_the_target_is_reached(tmp_path):
    assert _status_with(tmp_path, 100, [{"job_id": "1_0"}]).verdict == "done"


def test_status_unstarted_when_nothing_was_submitted(tmp_path):
    assert _status_with(tmp_path, 0, []).verdict == "unstarted"


def test_status_resumable_when_short_with_bins_on_disk(tmp_path):
    """A stopped chain holding bins is a restart, whatever SLURM called it."""
    got = _status_with(tmp_path, 40, [{"job_id": "1_0"}], "FAILED")
    assert got.verdict == "resumable"


def test_status_suspect_when_short_with_no_bins(tmp_path):
    """Zero bins after a run is a crash signature; requeuing it would loop.

    Its SLURM state is the same FAILED a wall-clock stop produces, which is
    exactly why the bin count and not the state decides.
    """
    got = _status_with(tmp_path, 0, [{"job_id": "1_0"}], "FAILED")
    assert got.verdict == "suspect"


def test_status_active_is_left_alone(tmp_path):
    got = _status_with(tmp_path, 10, [{"job_id": "1_0"}], "RUNNING")
    assert got.verdict == "active"
    assert got.active_job == "1_0"


# --- Campaign.status: what it is allowed *not* to read ----------------------
#
# A status check that re-reads every chain's data.h5 costs one HDF5 open per
# chain, which on a campaign-sized grid is the whole runtime. These pin the
# three tiers that avoid it -- and, just as importantly, the cases where the
# read must still happen.


def _status_counting_reads(tmp_path, chains, states=None, **kwargs):
    """Run status over a ledger of prebuilt chain records; report what it read."""
    led = _ledger(tmp_path)
    led.data["chains"].update(chains)
    led.save()

    read: list[str] = []

    def _counted(paths, *a, **k):
        read.extend(paths)
        return [0] * len(paths)

    camp = _campaign(tmp_path)
    with (
        patch(
            "py_alf.campaign.campaign.job_states",
            return_value=states or {},
        ),
        patch("py_alf.campaign.campaign.read_bin_counts", side_effect=_counted),
    ):
        return camp.status(Ledger.load(tmp_path / "c.json"), **kwargs), read


def _record(sim_dir, segments=(), **extra):
    return {"sim_dir": sim_dir, "point": {}, "segments": list(segments), **extra}


def test_status_never_reopens_a_finished_chain(tmp_path):
    """The cached count is trusted at the target: ALF only ever appends bins."""
    statuses, read = _status_counting_reads(
        tmp_path, {"a": _record(str(tmp_path / "sim"), bins=100)}
    )
    assert read == []
    assert statuses[0].bins == 100
    assert statuses[0].verdict == "done"


def test_status_re_reads_a_finished_chain_when_asked(tmp_path):
    """``deep`` is the escape hatch for data that changed under the ledger."""
    _, read = _status_counting_reads(
        tmp_path, {"a": _record(str(tmp_path / "sim"), bins=100)}, deep=True
    )
    assert read == [str(tmp_path / "sim" / "data.h5")]


def test_status_trusts_the_worker_record_for_an_idle_chain(tmp_path):
    """Nothing is running, so what the last segment flushed is what is on disk."""
    segments = [{"job_id": "1_0", "bins_after": 40}]
    statuses, read = _status_counting_reads(
        tmp_path,
        {"a": _record(str(tmp_path / "sim"), segments)},
        states={"1_0": {"status": "FAILED"}},
    )
    assert read == []
    assert statuses[0].bins == 40
    assert statuses[0].verdict == "resumable"


def test_status_reads_a_chain_whose_job_is_still_running(tmp_path):
    """A live job is writing bins the worker has not recorded yet."""
    segments = [{"job_id": "1_0", "bins_after": 40}]
    _, read = _status_counting_reads(
        tmp_path,
        {"a": _record(str(tmp_path / "sim"), segments)},
        states={"1_0": {"status": "RUNNING"}},
    )
    assert read == [str(tmp_path / "sim" / "data.h5")]


def test_status_reads_a_chain_whose_worker_left_no_record(tmp_path):
    """A segment killed before it could write one proves nothing about the file."""
    _, read = _status_counting_reads(
        tmp_path,
        {"a": _record(str(tmp_path / "sim"), [{"job_id": "1_0"}])},
        states={"1_0": {"status": "FAILED"}},
    )
    assert read == [str(tmp_path / "sim" / "data.h5")]


def test_status_caches_what_it_read_for_the_next_run(tmp_path):
    """The saved count is what makes the second check cheap."""
    led = _ledger(tmp_path)
    led.data["chains"]["a"] = _record(str(tmp_path / "sim"))
    led.save()
    camp = _campaign(tmp_path)
    with (
        patch("py_alf.campaign.campaign.job_states", return_value={}),
        patch(
            "py_alf.campaign.campaign.read_bin_counts",
            side_effect=lambda paths, *a, **k: [100] * len(paths),
        ),
    ):
        camp.status(Ledger.load(tmp_path / "c.json"))
    assert Ledger.load(tmp_path / "c.json").chains["a"]["bins"] == 100


def test_a_racing_read_cannot_walk_the_count_backwards(tmp_path):
    """A mid-write read comes back low; caching it would show lost progress."""
    led = _ledger(tmp_path)
    led.data["chains"]["a"] = _record(str(tmp_path / "sim"), bins=60)
    assert led.record_bins({"a": 80}) is True
    assert led.record_bins({"a": 0}) is False
    assert led.chains["a"]["bins"] == 80


def test_the_worker_record_taken_is_the_highest_one(tmp_path):
    """A requeued attempt that crashed early records fewer bins than it found."""
    segments = [
        {"job_id": "1_0", "bins_after": 40},
        {"job_id": "1_0", "bins_after": 12},
    ]
    statuses, read = _status_counting_reads(
        tmp_path,
        {"a": _record(str(tmp_path / "sim"), segments)},
        states={"1_0": {"status": "FAILED"}},
    )
    assert read == []
    assert statuses[0].bins == 40


# --- Campaign.status against real files, with nothing mocked ----------------


def _chain_on_disk(tmp_path, name, bins, job_id, worker_bins=None):
    """A chain directory holding real data, and the record its worker wrote."""
    import h5py
    import numpy as np

    d = tmp_path / name
    (d / "segments").mkdir(parents=True)
    with h5py.File(d / "data.h5", "w") as f:
        f.create_dataset("Ener_scal/obser", data=np.zeros((bins, 1)))
    if worker_bins is not None:
        (d / "segments" / "s0.json").write_text(
            json.dumps({"job_id": job_id, "bins_after": worker_bins})
        )
    return {
        "sim_dir": str(d),
        "point": {},
        "segments": [{"job_id": job_id}],
    }


def test_the_cached_answer_matches_what_a_full_read_would_say(tmp_path):
    """The property the whole optimisation rests on, checked without mocks.

    Every other status test stubs the reader out to observe *which* files get
    opened. That cannot catch a tier which is cheap but wrong, so this one puts
    real data.h5 files and real worker records on disk and requires the tiered
    answer to equal the one a full re-read gives -- on the first pass, on the
    cached second pass, and per chain rather than in aggregate.
    """
    chains = {
        "done": _chain_on_disk(tmp_path, "done", 100, "1_0", worker_bins=100),
        "idle": _chain_on_disk(tmp_path, "idle", 40, "1_1", worker_bins=40),
        "running": _chain_on_disk(tmp_path, "running", 55, "1_2", worker_bins=30),
        "crashed": _chain_on_disk(tmp_path, "crashed", 7, "1_3"),
        "unstarted": {
            "sim_dir": str(tmp_path / "nothing"),
            "point": {},
            "segments": [],
        },
    }
    led = _ledger(tmp_path)
    led.data["chains"].update(chains)
    led.save()

    states = {
        "1_0": {"status": "COMPLETED"},
        "1_1": {"status": "FAILED"},
        "1_2": {"status": "RUNNING"},
        "1_3": {"status": "FAILED"},
    }

    def run(deep):
        camp = _campaign(tmp_path)
        with patch("py_alf.campaign.campaign.job_states", return_value=states):
            st = camp.status(Ledger.load(tmp_path / "c.json"), deep=deep)
        return {s.chain_id: s.bins for s in st}

    truth = {"done": 100, "idle": 40, "running": 55, "crashed": 7, "unstarted": 0}
    assert run(deep=True) == truth
    assert run(deep=False) == truth, "a tier disagreed with the file on disk"
    assert run(deep=False) == truth, "the cached second pass disagreed"


def test_a_stale_worker_record_never_undercounts_a_running_chain(tmp_path):
    """The running chain above is the case the worker record would get wrong.

    Its last record says 30 bins; the file already holds 55. Reading it is
    exactly why a live job is excluded from the worker-record tier.
    """
    led = _ledger(tmp_path)
    led.data["chains"]["running"] = _chain_on_disk(
        tmp_path, "running", 55, "1_2", worker_bins=30
    )
    led.save()
    camp = _campaign(tmp_path)
    with patch(
        "py_alf.campaign.campaign.job_states",
        return_value={"1_2": {"status": "RUNNING"}},
    ):
        assert camp.status(Ledger.load(tmp_path / "c.json"))[0].bins == 55


def test_status_leaves_the_ledger_alone_when_told_not_to_persist(tmp_path):
    led = _ledger(tmp_path)
    led.data["chains"]["a"] = _chain_on_disk(tmp_path, "a", 100, "1_0", worker_bins=100)
    led.save()
    before = (tmp_path / "c.json").read_text()
    camp = _campaign(tmp_path)
    with patch("py_alf.campaign.campaign.job_states", return_value={}):
        assert camp.status(persist=False)[0].bins == 100
    assert (tmp_path / "c.json").read_text() == before


def test_absorbing_skips_only_the_chains_already_known_finished(tmp_path):
    """Skipping a scan must not skip a chain that could still gain records."""
    led = _ledger(tmp_path)
    led.data["chains"]["done"] = dict(
        _chain_on_disk(tmp_path, "done", 100, "1_0", worker_bins=100), bins=100
    )
    led.data["chains"]["short"] = dict(
        _chain_on_disk(tmp_path, "short", 40, "1_1", worker_bins=40), bins=40
    )
    assert led.absorb_segment_records(skip_finished=True) == 1
    assert led.chains["short"]["segments"][0]["bins_after"] == 40
    assert "bins_after" not in led.chains["done"]["segments"][0]
    assert led.absorb_segment_records() == 2


def test_record_bins_ignores_a_chain_the_ledger_does_not_hold(tmp_path):
    led = _ledger(tmp_path)
    led.data["chains"]["a"] = {"sim_dir": "/d", "point": {}, "segments": []}
    assert led.record_bins({"ghost": 50}) is False
    assert "ghost" not in led.chains


def test_chain_status_complete_tracks_the_target():
    base = ChainStatus(
        chain_id="a",
        sim_dir="/d",
        point={},
        bins=99,
        target_bins=100,
        segments=1,
        active_job=None,
        last_state=None,
        timed_out=False,
        verdict="resumable",
    )
    assert not base.complete
    assert replace(base, bins=100).complete


# --- caching an unfinished chain --------------------------------------------


def test_caching_an_unfinished_chain_agrees_with_a_full_read(tmp_path):
    """Same property as the finished-chain case, for idle unfinished chains."""
    chains = {
        "idle": _chain_on_disk(tmp_path, "idle", 40, "1_0"),
        "running": _chain_on_disk(tmp_path, "running", 55, "1_1"),
    }
    led = _ledger(tmp_path)
    led.data["chains"].update(chains)
    led.save()
    states = {"1_0": {"status": "FAILED"}, "1_1": {"status": "RUNNING"}}

    def run(deep):
        camp = _campaign(tmp_path)
        with patch("py_alf.campaign.campaign.job_states", return_value=states):
            return {s.chain_id: s.bins for s in camp.status(deep=deep)}

    truth = {"idle": 40, "running": 55}
    assert run(deep=False) == truth
    assert run(deep=False) == truth, "the cached second pass disagreed"
    assert run(deep=True) == truth


# --- job folders: one per array under jobs_dir ------------------------------


def _status_in_job_folders(tmp_path, segments, logs, state="FAILED"):
    """Status of one chain whose array logs live under ``jobs_dir/k/<array>``."""
    led = _ledger(tmp_path)
    led.data["chains"]["a"] = {
        "sim_dir": str(tmp_path / "sim"),
        "point": {},
        "array_key": "k",
        "segments": [{"job_id": j} for j in segments],
    }
    led.save()
    for job_id, text in logs.items():
        folder = tmp_path / "jobs" / "k" / job_id.split("_")[0]
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"{job_id}_0_log.out").write_text(text)

    camp = _campaign(tmp_path)
    camp.jobs_dir = tmp_path / "jobs"
    with (
        patch(
            "py_alf.campaign.campaign.job_states",
            return_value={segments[-1]: {"status": state}},
        ),
        patch(
            "py_alf.campaign.campaign.read_bin_counts",
            side_effect=lambda paths, *a, **k: [10] * len(paths),
        ),
    ):
        return camp.status(Ledger.load(tmp_path / "c.json"))[0]


def test_status_reads_a_requeued_jobs_log_from_its_array_folder(tmp_path):
    """A requeue keeps its job id, so its log stays in the same array folder."""
    got = _status_in_job_folders(
        tmp_path, ["5101_0"], {"5101_0": "... this job is timed-out ..."}
    )
    assert got.timed_out
    assert got.last_state == "TIMEOUT"


def test_status_reads_the_resubmitted_arrays_log_not_the_first(tmp_path):
    """reconcile's new array has a new id, hence its own folder."""
    got = _status_in_job_folders(
        tmp_path,
        ["5201_0", "5202_0"],
        {"5201_0": "... this job is timed-out ...", "5202_0": "Traceback ..."},
    )
    assert not got.timed_out
    assert got.last_state == "FAILED"


def test_launch_submits_each_array_into_its_own_folder_template(tmp_path):
    chains = [_chain(tmp_path, "a", 0, array_key="L8"), _chain(tmp_path, "b", 0)]
    camp, sub = _launch_campaign(tmp_path, chains, jobs_dir=tmp_path / "jobs")
    camp.launch(verbose=False)

    assert [c["submit_dir"] for c in sub.calls] == [
        tmp_path / "jobs" / "L8" / "%A",
        tmp_path / "jobs" / "k" / "%A",
    ]
    assert camp.job_folder("L8", "1001_0") == tmp_path / "jobs" / "L8" / "1001"


def test_without_jobs_dir_files_stay_in_the_submitters_folder(tmp_path):
    camp, sub = _launch_campaign(tmp_path, [_chain(tmp_path, "a", 0)])
    camp.launch(verbose=False)

    assert sub.calls[0]["submit_dir"] is None
    assert camp.job_folder("k", "1001_0") == tmp_path / "submit"


# --- housekeeping: log_path, cancel, prune -----------------------------------


def _housekeeping_campaign(tmp_path, segments_by_chain, executor="slurm"):
    """A campaign under jobs_dir whose chains ran the given segments in array k."""
    led = _ledger(tmp_path)
    for cid, jobs in segments_by_chain.items():
        led.data["chains"][cid] = {
            "sim_dir": str(tmp_path / cid),
            "point": {},
            "array_key": "k",
            "segments": [{"job_id": j} for j in jobs],
        }
        for j in jobs:
            folder = tmp_path / "jobs" / "k" / j.split("_")[0]
            folder.mkdir(parents=True, exist_ok=True)
            (folder / f"{j}_0_log.out").write_text("log")
            (folder / f"{j}_submitted.pkl").write_text("pickle")
    led.save()
    camp = _campaign(tmp_path)
    camp.submitter.executor = executor
    camp.jobs_dir = tmp_path / "jobs"
    return camp


def test_log_path_is_the_latest_segments_log(tmp_path):
    camp = _housekeeping_campaign(tmp_path, {"a": ["601_0", "602_3"], "b": []})
    assert camp.log_path("a") == tmp_path / "jobs" / "k" / "602" / "602_3_0_log.out"
    assert camp.log_path("a", "err").name == "602_3_0_log.err"
    assert camp.log_path("b") is None


def test_cancel_cancels_each_queued_array_of_the_campaign(tmp_path):
    """Arrays come from the ledger and the job folders; others' jobs are not ours."""
    camp = _housekeeping_campaign(
        tmp_path, {"a": ["701_0"], "b": ["701_1"], "c": ["700_0"]}
    )
    (tmp_path / "jobs" / "k" / "705").mkdir()  # submitted, never recorded
    queued = {"701", "705", "999"}  # 999 belongs to someone else's work
    with (
        patch("py_alf.campaign.campaign.queued_arrays", return_value=queued),
        patch("py_alf.campaign.campaign.cancel") as scancel,
    ):
        assert camp.cancel(dry_run=True) == ["701", "705"]
        scancel.assert_not_called()
        assert camp.cancel() == ["701", "705"]
    scancel.assert_called_once_with(["701", "705"])


def test_prune_keeps_what_is_still_read(tmp_path):
    """Superseded arrays go; the latest keeps its log; a queued array is untouched."""
    camp = _housekeeping_campaign(
        tmp_path, {"a": ["801_0", "802_0"], "b": ["803_0"], "c": ["804_0"]}
    )
    jobs = tmp_path / "jobs" / "k"
    with patch("py_alf.campaign.campaign.queued_arrays", return_value={"803"}):
        planned = camp.prune(dry_run=True)
        assert (jobs / "801").exists()
        assert camp.prune() == planned

    assert not (jobs / "801").exists()  # superseded by 802 for chain a
    assert sorted(p.name for p in (jobs / "802").iterdir()) == ["802_0_0_log.out"]
    assert (jobs / "803" / "803_0_submitted.pkl").exists()  # queued
    assert sorted(p.name for p in (jobs / "804").iterdir()) == ["804_0_0_log.out"]


def test_prune_spares_a_queued_array_the_ledger_never_recorded(tmp_path):
    """A launch that failed partway can leave a live array out of the ledger."""
    camp = _housekeeping_campaign(tmp_path, {"a": ["901_0"]})
    unrecorded = tmp_path / "jobs" / "k" / "902"
    unrecorded.mkdir()
    (unrecorded / "902_0_submitted.pkl").write_text("pickle")
    with patch("py_alf.campaign.campaign.queued_arrays", return_value={"902"}):
        camp.prune()
    assert (unrecorded / "902_0_submitted.pkl").exists()


def test_prune_refuses_when_squeue_fails(tmp_path):
    """An unanswered query must not read as an empty queue."""
    camp = _housekeeping_campaign(tmp_path, {"a": ["911_0", "912_0"]})
    failure = subprocess.CalledProcessError(1, "squeue")
    with (
        patch("py_alf.campaign.campaign.queued_arrays", side_effect=failure),
        pytest.raises(RuntimeError, match="squeue failed"),
    ):
        camp.prune()
    assert (tmp_path / "jobs" / "k" / "911").exists()


def test_launch_refuses_when_slurm_does_not_answer(tmp_path):
    """Treating an ERROR state as idle would submit a running chain again."""
    camp, sub = _launch_campaign(tmp_path, [_chain(tmp_path, "a", 0)], executor="slurm")
    camp.launch(verbose=False)
    with (
        patch(
            "py_alf.campaign.campaign.job_states",
            return_value={"1001_0": {"status": "ERROR"}},
        ),
        pytest.raises(RuntimeError, match="did not answer"),
    ):
        camp.launch(verbose=False)
    assert len(sub.calls) == 1


def test_launch_records_each_array_before_submitting_the_next(tmp_path):
    """A failure on a later array must leave the earlier ones in the ledger."""
    chains = [
        _chain(tmp_path, "a", 0, array_key="k1"),
        _chain(tmp_path, "b", 0, array_key="k2"),
    ]
    camp, sub = _launch_campaign(tmp_path, chains)
    real_submit = sub.submit

    def submit_then_fail(sims, job_properties, **kwargs):
        if sub.calls:
            raise RuntimeError("sbatch: error")
        return real_submit(sims, job_properties, **kwargs)

    sub.submit = submit_then_fail
    with pytest.raises(RuntimeError, match="sbatch"):
        camp.launch(verbose=False)

    ledger = Ledger.load(tmp_path / "c.json")
    assert len(ledger.chains["a"]["segments"]) == 1


def test_dry_run_survives_arrays_that_already_hold_bins(tmp_path):
    """The dry-run listing once overwrote launch's own known-bins argument."""
    chains = [
        _chain(tmp_path, "a", 20, array_key="k1"),
        _chain(tmp_path, "b", 30, array_key="k2"),
    ]
    camp, sub = _launch_campaign(tmp_path, chains)
    camp.launch(dry_run=True, verbose=False, known_bins={"a": 20, "b": 30})
    camp.launch(dry_run=True, verbose=False)
    assert sub.calls == []


def test_prune_without_jobs_dir_touches_nothing(tmp_path):
    camp = _campaign(tmp_path)
    assert camp.prune() == []


# --- the progress hook -------------------------------------------------------


def test_status_reports_every_chain_to_the_progress_hook_exactly_once(tmp_path):
    """A bar can only reach 100% if every tier accounts for the chains it took.

    Each tier settles a different subset, so a tier that resolved chains without
    reporting them would leave the caller's bar stuck short of the total for the
    whole run -- and the miscount would scale with how much of the grid was in
    that tier, which is exactly the state that varies over a campaign's life.
    """
    led = _ledger(tmp_path)
    led.data["chains"] = {
        "cached": dict(_chain_on_disk(tmp_path, "cached", 100, "1_0"), bins=100),
        "worker": _chain_on_disk(tmp_path, "worker", 40, "1_1", worker_bins=40),
        "read": _chain_on_disk(tmp_path, "read", 55, "1_2", worker_bins=30),
        "unstarted": {"sim_dir": str(tmp_path / "no"), "point": {}, "segments": []},
    }
    led.save()
    states = {"1_1": {"status": "FAILED"}, "1_2": {"status": "RUNNING"}}

    seen: list[tuple[int, str]] = []
    camp = _campaign(tmp_path)
    with patch("py_alf.campaign.campaign.job_states", return_value=states):
        statuses = camp.status(
            Ledger.load(tmp_path / "c.json"),
            on_progress=lambda n, p: seen.append((n, p)),
        )

    assert sum(n for n, _ in seen) == len(statuses) == 4
    # Phases are announced even when they settle nothing, so a bar shows what it
    # is waiting on rather than looking hung during the scan.
    assert {"scanning", "cached", "reading"} <= {phase for _, phase in seen}


def test_the_progress_hook_is_optional(tmp_path):
    """Nothing reports unless a hook is passed; the core owns no bar."""
    led = _ledger(tmp_path)
    led.data["chains"]["a"] = _chain_on_disk(tmp_path, "a", 100, "1_0", worker_bins=100)
    led.save()
    camp = _campaign(tmp_path)
    with patch("py_alf.campaign.campaign.job_states", return_value={}):
        assert camp.status()[0].bins == 100  # must not raise


# --- Campaign.launch and Campaign.reconcile ---------------------------------
#
# What these guard is the pairing between a chain and the bin count that sizes
# its budget. `_runnable` returns them together precisely so the launch does not
# re-read the grid, and a misalignment there would hand every chain another
# chain's budget while every job still submitted and every test still passed.


class _FakeSubmitter:
    """Stands in for ClusterSubmitter, recording what a launch asked for.

    Like ``submit``, writes each job's id into its chain's own ``jobid.txt``,
    which is how ``_submit_array`` pairs chains with jobs.
    """

    def __init__(self, submit_dir, executor="local"):
        self.submit_dir = submit_dir
        self.executor = executor
        self.partition_rules = RULES
        self.calls = []
        self._array = 1000

    def submit(self, sims, job_properties, **kwargs):
        self._array += 1
        self.calls.append(
            {"sims": list(sims), "job_properties": dict(job_properties), **kwargs}
        )
        jobs = []
        for i, sim in enumerate(sims):
            job_id = f"{self._array}_{i}"
            Path(sim.sim_dir).mkdir(parents=True, exist_ok=True)
            (Path(sim.sim_dir) / "jobid.txt").write_text(job_id)
            jobs.append(SimpleNamespace(job_id=job_id))
        return jobs


def _chain(tmp_path, name, bins, array_key="k", target_bins=100):
    """A Chain whose data.h5 really holds ``bins`` bins."""
    import h5py
    import numpy as np

    d = tmp_path / name
    d.mkdir(parents=True, exist_ok=True)
    with h5py.File(d / "data.h5", "w") as f:
        f.create_dataset("Ener_scal/obser", data=np.zeros((bins, 1)))
    sim = _sim(d)
    sim.sim_dict = {}
    sim.mc_seed = 1
    return Chain(
        chain_id=name,
        sim=sim,
        target_bins=target_bins,
        array_key=array_key,
    )


def _launch_campaign(tmp_path, chains, **kwargs):
    submitter = _FakeSubmitter(
        tmp_path / "submit", executor=kwargs.pop("executor", "local")
    )
    camp = Campaign(
        name="c",
        chains=list(chains),
        target_bins=100,
        submitter=submitter,
        ledger_path=tmp_path / "c.json",
        # Pin the cost so a budget is a function of the bins alone.
        cost_model=lambda sim_dict: 0.1,
        **kwargs,
    )
    return camp, submitter


def test_launching_an_empty_grid_is_a_clean_error(tmp_path):
    camp, sub = _launch_campaign(tmp_path, [])
    with pytest.raises(SystemExit):
        camp.launch(verbose=False)
    assert sub.calls == []


def test_launch_submits_only_the_chains_short_of_the_target(tmp_path):
    chains = [_chain(tmp_path, "done", 100), _chain(tmp_path, "short", 20)]
    camp, sub = _launch_campaign(tmp_path, chains)
    camp.launch(verbose=False)

    assert [s.sim_dir for s in sub.calls[0]["sims"]] == [chains[1].sim_dir]
    ledger = Ledger.load(tmp_path / "c.json")
    assert ledger.chains["short"]["segments"][0]["job_id"] == "1001_0"
    assert ledger.chains["done"]["segments"] == []


def test_launch_sizes_each_array_by_the_work_that_array_has_left(tmp_path):
    """The pairing test: a nearly-done chain must not inherit an empty one's budget.

    Two arrays, one chain each, differing only in bins already on disk. If the
    counts and the chains came apart, the budgets would simply swap -- both
    arrays would still submit, and nothing else here would notice.
    """
    chains = [
        _chain(tmp_path, "fresh", 0, array_key="a"),
        _chain(tmp_path, "nearly", 99, array_key="b"),
    ]
    camp, sub = _launch_campaign(tmp_path, chains)
    camp.launch(verbose=False)

    budget = {
        call["sims"][0].sim_dir: call["sims"][0].sim_dict["CPU_MAX"]
        for call in sub.calls
    }
    assert budget[chains[0].sim_dir] > budget[chains[1].sim_dir]
    # 1 bin left at 0.1 h/bin, against 100 -- the floor is all the second needs.
    assert budget[chains[1].sim_dir] == pytest.approx(camp.policy.min_hours)


def test_launch_groups_one_array_per_array_key(tmp_path):
    chains = [
        _chain(tmp_path, "a1", 0, array_key="a"),
        _chain(tmp_path, "a2", 0, array_key="a"),
        _chain(tmp_path, "b1", 0, array_key="b"),
    ]
    camp, sub = _launch_campaign(tmp_path, chains)
    camp.launch(verbose=False)

    assert [len(c["sims"]) for c in sub.calls] == [2, 1]


def test_launch_leaves_out_a_chain_whose_job_is_still_active(tmp_path):
    """One bulk query decides; submit is then told not to check again."""
    chains = [_chain(tmp_path, "held", 10), _chain(tmp_path, "free", 10)]
    camp, sub = _launch_campaign(tmp_path, chains, executor="slurm")
    camp.launch(verbose=False)  # one array: held is job 1001_0, free is 1001_1
    states = {"1001_0": {"status": "RUNNING"}, "1001_1": {"status": "COMPLETED"}}

    with patch("py_alf.campaign.campaign.job_states", return_value=states) as query:
        camp.launch(verbose=False)

    query.assert_called_once()
    assert [s.sim_dir for s in sub.calls[1]["sims"]] == [chains[1].sim_dir]
    assert sub.calls[1]["skip_active"] is False
    ledger = Ledger.load(tmp_path / "c.json")
    assert len(ledger.chains["held"]["segments"]) == 1
    assert len(ledger.chains["free"]["segments"]) == 2


def test_launch_hands_the_array_its_requeue_budget(tmp_path):
    """The attempt count reaches submit as max_requeues, not a magic job property."""
    camp, sub = _launch_campaign(tmp_path, [_chain(tmp_path, "a", 0)])
    camp.launch(segments=4, verbose=False)

    assert sub.calls[0]["max_requeues"] == 4
    assert "slurm_max_num_timeout" not in sub.calls[0]["job_properties"]


def test_launch_caps_the_budget_with_the_submitter_partition_rules(tmp_path):
    """The policy reads the limits the submitter was built with, not a copy."""
    camp, sub = _launch_campaign(tmp_path, [_chain(tmp_path, "a", 0)])
    sub.partition_rules = {"medium": {"max_hours": 10.0}}
    camp.launch(verbose=False)  # 100 bins at 0.1 h would want 13 h

    assert sub.calls[0]["sims"][0].sim_dict["CPU_MAX"] == pytest.approx(10.0 * 0.95)


def test_launch_dry_run_submits_nothing_and_writes_no_ledger(tmp_path):
    camp, sub = _launch_campaign(tmp_path, [_chain(tmp_path, "a", 0)])
    camp.launch(dry_run=True, verbose=False)

    assert sub.calls == []
    assert not (tmp_path / "c.json").exists()


def test_relaunching_keeps_the_history_of_a_chain_already_run(tmp_path):
    """Topping up a campaign must extend a chain's record, not reset it."""
    camp, _ = _launch_campaign(tmp_path, [_chain(tmp_path, "a", 0)])
    camp.launch(verbose=False)
    camp.launch(verbose=False)

    assert len(Ledger.load(tmp_path / "c.json").chains["a"]["segments"]) == 2


def _reconcile_campaign(tmp_path, specs, **kwargs):
    """Build a launched campaign whose chains sit in the given states.

    ``specs`` maps a name to ``(bins, slurm_state)``; ``None`` means the chain
    was never submitted at all.
    """
    chains = [_chain(tmp_path, name, bins) for name, (bins, _) in specs.items()]
    camp, sub = _launch_campaign(tmp_path, chains, **kwargs)
    led = _ledger(tmp_path)
    states = {}
    for i, (name, (_, state)) in enumerate(specs.items()):
        segments = []
        if state is not None:
            job_id = f"900_{i}"
            segments = [{"job_id": job_id}]
            states[job_id] = {"status": state}
        led.data["chains"][name] = {
            "sim_dir": str(tmp_path / name),
            "point": {},
            "segments": segments,
        }
    led.save()
    return camp, sub, states


def test_reconcile_resubmits_only_what_stalled(tmp_path):
    """done and active are left alone; a stopped chain holding bins is resumed."""
    camp, sub, states = _reconcile_campaign(
        tmp_path,
        {
            "done": (100, "COMPLETED"),
            "active": (30, "RUNNING"),
            "resumable": (40, "FAILED"),
            "unstarted": (0, None),
        },
    )
    with patch("py_alf.campaign.campaign.job_states", return_value=states):
        camp.reconcile(verbose=False)

    resubmitted = {s.sim_dir for call in sub.calls for s in call["sims"]}
    assert resubmitted == {str(tmp_path / "resumable"), str(tmp_path / "unstarted")}


def test_reconcile_reads_each_data_file_at_most_once(tmp_path):
    """launch reuses the counts status just resolved instead of re-reading."""
    camp, sub, states = _reconcile_campaign(
        tmp_path,
        {"resumable": (40, "FAILED"), "unstarted": (0, None)},
    )
    reads: list[str] = []

    def _counted(paths, *a, **k):
        reads.extend(paths)
        return read_bin_counts(paths, *a, **k)

    with (
        patch("py_alf.campaign.campaign.job_states", return_value=states),
        patch("py_alf.campaign.campaign.read_bin_counts", side_effect=_counted),
    ):
        camp.reconcile(verbose=False)

    assert len(reads) == len(set(reads))
    assert {s.sim_dir for call in sub.calls for s in call["sims"]} == {
        str(tmp_path / "resumable"),
        str(tmp_path / "unstarted"),
    }


def test_reconcile_leaves_a_suspect_chain_alone_until_forced(tmp_path):
    """Zero bins after a run is a crash signature; requeueing it would loop."""
    specs = {"suspect": (0, "FAILED")}
    camp, sub, states = _reconcile_campaign(tmp_path, specs)
    with patch("py_alf.campaign.campaign.job_states", return_value=states):
        camp.reconcile(verbose=False)
    assert sub.calls == []

    camp, sub, states = _reconcile_campaign(tmp_path, specs)
    with patch("py_alf.campaign.campaign.job_states", return_value=states):
        camp.reconcile(force=True, verbose=False)
    assert [s.sim_dir for s in sub.calls[0]["sims"]] == [str(tmp_path / "suspect")]


def test_reconcile_reports_without_submitting_when_asked(tmp_path):
    camp, sub, states = _reconcile_campaign(tmp_path, {"resumable": (40, "FAILED")})
    with patch("py_alf.campaign.campaign.job_states", return_value=states):
        statuses = camp.reconcile(submit=False, verbose=False)
    assert sub.calls == []
    assert [s.verdict for s in statuses] == ["resumable"]


def test_reconcile_persists_the_worker_records_it_absorbed(tmp_path):
    """reconcile stopped absorbing and saving itself; status must still do both.

    Dropping those two calls is only safe because status() now folds the worker
    records in and writes the ledger. If it ever stops, the elapsed times and
    bin counts a node reported would be silently lost on every reconcile.
    """
    camp, sub, states = _reconcile_campaign(tmp_path, {"resumable": (40, "FAILED")})
    seg = Path(tmp_path / "resumable" / "segments")
    seg.mkdir(parents=True)
    (seg / "000-900_0.json").write_text(
        json.dumps({"job_id": "900_0", "bins_after": 40, "elapsed_s": 1234})
    )
    with patch("py_alf.campaign.campaign.job_states", return_value=states):
        camp.reconcile(submit=False, verbose=False)

    on_disk = Ledger.load(tmp_path / "c.json").chains["resumable"]
    assert on_disk["segments"][0]["elapsed_s"] == 1234
    assert on_disk["bins"] == 40
