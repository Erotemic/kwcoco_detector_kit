"""
Optional line-level profiling hook.

Decorate hot functions with ``@profile`` (imported from here). By default
this is the no-op ``line_profiler`` global decorator — near-zero overhead.
Enable at runtime by setting ``LINE_PROFILE=1`` in the environment;
line_profiler then profiles every decorated function and, on interpreter
exit, dumps ``profile_output.txt`` + ``profile_output.lprof`` to the CWD
(so run with the CWD bind-mounted to a host path to keep the output).

Falls back to a plain no-op if line_profiler isn't installed, so the kit
never hard-depends on it.
"""
try:  # line_profiler>=4: `profile` is env-gated (LINE_PROFILE), ~free when off
    from line_profiler import profile  # noqa: F401
except Exception:  # pragma: no cover - line_profiler absent
    def profile(func):
        return func
