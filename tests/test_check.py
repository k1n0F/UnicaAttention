"""unica.compatibility() on machines this one is not: every reason it says ok, untested or cannot run."""
from unica.check import OK, UNSUPPORTED, UNTESTED, compatibility

TESTED = {"platform": "win32", "torch": "2.14.0+cu132", "torch_cuda": "13.2", "gpu": "NVIDIA GeForce RTX 5080",
          "capability": (12, 0), "driver": 13040}


def judge(**changes):
    return compatibility(TESTED | changes)


def test_the_tested_machine_is_ok():
    assert judge() == (OK, ())
    assert judge(driver=13050).status == OK  # a newer driver runs older cubins


def test_untested_but_runs():
    for changes in ({"gpu": "NVIDIA GeForce RTX 5090"}, {"gpu": "NVIDIA RTX PRO 6000 Blackwell Workstation Edition"},
                    {"driver": 13020}, {"torch": "2.15.0+cu132"}, {"platform": "freebsd"}):
        status = judge(**changes)
        assert status.status == UNTESTED and len(status.reasons) == 1, changes
    assert len(judge(platform="freebsd", gpu="NVIDIA GeForce RTX 5060").reasons) == 2
    assert judge(platform="linux").status == OK  # WSL2 on the RTX 5080 (LOG M4.29)


def test_cannot_run():
    for changes in ({"capability": (12, 1), "gpu": "NVIDIA GB10"}, {"capability": (10, 0)}, {"capability": (8, 9)},
                    {"driver": 12080}, {"driver": None}, {"torch_cuda": None, "capability": None, "gpu": None},
                    {"capability": None, "gpu": None}):
        status = judge(**changes)
        assert status.status == UNSUPPORTED and status.reasons, changes
    # what cannot run is all that is said, not what is merely untested besides
    assert judge(capability=(8, 9), platform="freebsd").reasons == judge(capability=(8, 9)).reasons
