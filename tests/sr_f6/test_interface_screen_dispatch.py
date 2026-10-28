import pytest

from sr_f6.interface_screen_dispatch import QOS, _gpu, capacity


def query_for(jobs, *, slots="5", gpu_limit="8"):
    def query(args):
        if args[0] == "sacctmgr":
            return f"{QOS}|{slots}|gres/gpu:pro6000={gpu_limit}|"
        if args[0] == "squeue":
            assert args[3] == QOS
            return "\n".join(f"{j[0]}|{j[1]}|example|{j[2]}|{QOS}" for j in jobs)
        row = next(j for j in jobs if j[0] == args[-1])
        return f"JobId={row[0]} QOS={QOS} ReqTRES={row[3]} AllocTRES={row[4]}"

    return query


def job(i, gpu=1, user="varun024", state="RUNNING", allocated=None):
    return (
        str(i),
        user,
        state,
        f"cpu=4,gres/gpu={gpu},gres/gpu:pro6000={gpu}",
        "" if allocated is None else f"gres/gpu={allocated}",
    )


def test_pending_and_other_users_reserve_gpu_but_not_user_slots():
    result = capacity(
        query_for(
            [
                job(1, 2),
                job(2, 2, state="PENDING"),
                job(3, 3, user="someone_else"),
            ]
        )
    )
    assert result["reserved_gpus"] == 7
    assert result["submitted_by_user"] == 2
    assert result["can_submit_two_gpu_job"] is False


def test_allocated_larger_than_request_and_unknown_state_are_counted():
    result = capacity(query_for([job(1, 1, allocated=4, state="COMPLETING"), job(2, 2)]))
    assert result["reserved_gpus"] == 6
    assert result["can_submit_two_gpu_job"] is True


def test_five_slots_prevent_even_small_gpu_jobs():
    result = capacity(query_for([job(i) for i in range(5)]))
    assert result["reserved_gpus"] == 5
    assert result["can_submit_two_gpu_job"] is False


@pytest.mark.parametrize("tres", ["", "cpu=4", "gres/gpu=N/A", "gres/gpu:a40=1"])
def test_unknown_reservations_fail_closed(tres):
    with pytest.raises(PermissionError):
        _gpu(tres)


@pytest.mark.parametrize("slots,gpus", [("6", "8"), ("5", "10"), ("", "8")])
def test_changed_limits_cannot_expand_authorization(slots, gpus):
    with pytest.raises(PermissionError):
        capacity(query_for([], slots=slots, gpu_limit=gpus))


def test_duplicate_or_array_queue_ids_fail_closed():
    for jobs in ([job(1), job(1)], [job("1_2")]):
        with pytest.raises(PermissionError):
            capacity(query_for(jobs))


def test_live_capacity_empty_and_three_jobs():
    assert capacity(query_for([]))["can_submit_two_gpu_job"] is True
    assert capacity(query_for([job(i) for i in range(3)]))["can_submit_two_gpu_job"] is True
