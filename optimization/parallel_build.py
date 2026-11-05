"""Build concurrently, then hand each unchanged receipt to its native loader once."""
import copy
import os
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor


def install(jobs):
    # Jobs are (module, output-directory); import modules before starting threads.
    env = dict(os.environ)
    originals = {name: module.ensure_binary for name, (module, _) in jobs.items()}
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        pending = {name: pool.submit(originals[name], folder)
                   for name, (_, folder) in jobs.items()}
        results = {name: future.result() for name, future in pending.items()}
    assert dict(os.environ) == env
    stats = dict(enabled=True, wall_seconds=time.monotonic()-started, restored=False,
                 builds={name: dict(binary=str(value[0]), receipt=copy.deepcopy(value[1]),
                                    timing=copy.deepcopy(value[2]), consumed=False)
                         for name, value in results.items()})
    wrappers = {}
    for name, (module, folder) in jobs.items():
        def ensure(output, name=name, folder=Path(folder).resolve()):
            assert Path(output).resolve() == folder, ('Unexpected output', name)
            assert not stats['builds'][name]['consumed'], ('Duplicate load', name)
            stats['builds'][name]['consumed'] = True
            # Native installers may append their call counters to timing dicts.
            return copy.deepcopy(results[name])
        wrappers[name] = ensure
        module.ensure_binary = ensure
    def restore():
        for name, (module, _) in jobs.items():
            assert module.ensure_binary is wrappers[name]
        for name, (module, _) in jobs.items():
            module.ensure_binary = originals[name]
        stats['restored'] = True
    return restore, stats
