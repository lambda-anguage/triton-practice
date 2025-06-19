from torch.utils import benchmark

def bench(fn, *args, **kwargs):
    timer = benchmark.Timer(
        stmt="fn(*args, **kwargs)", globals={"fn": fn, "args": args, "kwargs": kwargs}
    )
    return timer.blocked_autorange(min_run_time=2.).mean
