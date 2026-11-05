import ctypes, json, os, signal, sys, time
from pathlib import Path
OWNERSHIP_ENV = "PROTENIX_RESEARCHBOOST_LAUNCH_NONCE"

def _pidfd_syscall(number, *args):
    # Some task Python builds omit pidfd wrappers even on supported Linux kernels.
    # Linux x86_64 UAPI numbers; this fallback is restricted to the verified task host.
    if sys.platform != "linux" or os.uname().machine != "x86_64":
        raise RuntimeError("No supported safe pidfd implementation on this platform")
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    result = libc.syscall(ctypes.c_long(number), *args)
    if result < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return result

def open_pidfd(pid):
    if hasattr(os, "pidfd_open"):
        return os.pidfd_open(pid)
    return _pidfd_syscall(434, ctypes.c_int(pid), ctypes.c_uint(0))

def send_pidfd(fd, sig):
    if hasattr(signal, "pidfd_send_signal"):
        return signal.pidfd_send_signal(fd, sig)
    return _pidfd_syscall(424, ctypes.c_int(fd), ctypes.c_int(sig), ctypes.c_void_p(), ctypes.c_uint(0))

def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")

class OwnedProcesses:
    """Own only an exact per-launch environment nonce, regardless of ancestry/session."""
    def __init__(self, nonce, output):
        self.nonce, self.output = nonce, Path(output)
        self.marker = f"{OWNERSHIP_ENV}={nonce}".encode()
        self.events, self.errors, self.observed = [], [], {}

    @staticmethod
    def supported():
        if not Path("/proc/self/stat").is_file():
            raise RuntimeError("Owned cleanup requires Linux /proc and pidfd support before launching")
        fd = open_pidfd(os.getpid())
        try:
            send_pidfd(fd, 0)  # Capability/permission check only; signal zero changes no process state.
        finally:
            os.close(fd)

    def inspect(self, pid):
        if pid == os.getpid():
            return None  # The nonce is never placed in the evaluator's environment.
        directory = Path("/proc") / str(pid)
        try:
            first = (directory / "stat").read_text()
            fields = first[first.rfind(")") + 2:].split()
            starttime = int(fields[19])
            if self.marker not in (directory / "environ").read_bytes().split(b"\0"):
                return None
            second = (directory / "stat").read_text()
            current = second[second.rfind(")") + 2:].split()
            if int(current[19]) != starttime or current[0] == "Z":
                return None
            return {"pid": pid, "starttime_ticks": starttime, "ppid": int(current[1]),
                    "pgid": int(current[2]), "session": int(current[3])}
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            return None

    def scan(self):
        found = []
        for path in Path("/proc").iterdir():
            if path.name.isdigit():
                item = self.inspect(int(path.name))
                if item is not None:
                    found.append(item)
                    self.observed[(item["pid"], item["starttime_ticks"])] = item
        return found

    def send(self, identity, sig):
        pid, expected = identity["pid"], identity["starttime_ticks"]
        fd = None
        try:
            # pidfd pins the process identity across PID reuse; recheck marker/starttime after opening.
            fd = open_pidfd(pid)
            current = self.inspect(pid)
            if current is None or current["starttime_ticks"] != expected:
                self.events.append({"pid": pid, "starttime_ticks": expected, "signal": sig.name,
                                    "action": "skipped_identity_changed", "wall_time_ns": time.time_ns()})
                return
            send_pidfd(fd, sig)
            self.events.append({**current, "signal": sig.name, "action": "sent", "transport": "pidfd",
                                "wall_time_ns": time.time_ns()})
        except ProcessLookupError:
            pass
        except OSError as error:
            self.errors.append({"pid": pid, "signal": sig.name, "error": f"{type(error).__name__}: {error}"})
        finally:
            if fd is not None:
                os.close(fd)

    def cleanup(self, process=None, reason="finalize", term_seconds=5., kill_seconds=5.):
        started, clean = time.monotonic(), False
        try:
            for sig, seconds in ((signal.SIGTERM, term_seconds), (signal.SIGKILL, kill_seconds)):
                deadline, sent, empty = time.monotonic() + seconds, set(), 0
                while True:
                    if process is not None:
                        process.poll()  # Reap our direct child; reparented zombies hold no live resources.
                    found = self.scan()
                    if not found:
                        empty += 1
                        if empty >= 2:
                            clean = True
                            break
                    else:
                        empty = 0
                        for item in found:
                            key = (item["pid"], item["starttime_ticks"])
                            if key not in sent:
                                self.send(item, sig)
                                sent.add(key)
                    if time.monotonic() >= deadline:
                        break
                    time.sleep(min(.1, max(0, deadline - time.monotonic())))
                if clean:
                    break
            survivors = self.scan()
            clean = clean and not survivors and not self.errors
        except Exception as error:
            self.errors.append({"error": f"{type(error).__name__}: {error}"})
            survivors, clean = None, False
        report = {"schema": "protenix-owned-cleanup@1", "status": "CLEAN" if clean else "ERROR",
                  "marker_name": OWNERSHIP_ENV, "nonce": self.nonce, "reason": reason,
                  "root_pid": process.pid if process is not None else None,
                  "policy": {"term_seconds": term_seconds, "kill_seconds": kill_seconds, "signal_transport": "pidfd"},
                  "observed": list(self.observed.values()), "events": self.events, "errors": self.errors,
                  "survivors": survivors, "elapsed_seconds": time.monotonic() - started,
                  "finished_at_ns": time.time_ns()}
        write(self.output, report)
        if not clean:
            raise RuntimeError("Owned subprocess cleanup incomplete; inspect cleanup.json")
        return report
