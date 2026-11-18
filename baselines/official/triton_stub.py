"""Stand-in for the `triton` package so the official fla / mamba_ssm Python files import on a
Mac CPU. No Triton kernel is ever run: every kernel call used by a baseline is replaced by the
official naive PyTorch path (see baselines/official/load.py). Calling a stubbed kernel raises."""
import sys
import types


class _Any:
    def __init__(self, name="triton"):
        self._name = name

    def __getattr__(self, k):
        if k.startswith("__"):
            raise AttributeError(k)
        return _Any(f"{self._name}.{k}")

    def __call__(self, *a, **k):
        if len(a) == 1 and callable(a[0]) and not k:          # used as a decorator
            return _Kernel(a[0])
        return _Deco()

    def __getitem__(self, k):
        return self

    def __or__(self, o):
        return self

    __ror__ = __or__

    def __mro_entries__(self, bases):
        return (_Base,)


class _Base:
    def __init__(self, *a, **k):
        pass

    def __getattr__(self, k):
        if k.startswith("__"):
            raise AttributeError(k)
        return _Any(k)


class _Deco:
    def __call__(self, f):
        return _Kernel(f)


class _Kernel:
    def __init__(self, f):
        import inspect
        self.fn = f
        self.__name__ = getattr(f, "__name__", "kernel")
        try:
            self.arg_names = list(inspect.signature(f).parameters)
        except (TypeError, ValueError):
            self.arg_names = []

    @property
    def configs(self):                       # module-level code reads kernel.configs[i].kwargs[...]
        class _Kw(dict):
            def __missing__(self, k):
                return 16
        return [types.SimpleNamespace(kwargs=_Kw(), num_warps=4, num_stages=1)]

    def __getitem__(self, grid):
        def run(*a, **k):
            raise RuntimeError(f"Triton kernel {self.__name__} called on CPU; use the naive path")
        return run

    def __call__(self, *a, **k):
        return self.fn(*a, **k)


class _Mod(types.ModuleType):
    def __getattr__(self, k):
        if k.startswith("__"):
            raise AttributeError(k)
        return _Any(f"{self.__name__}.{k}")


def install():
    if "triton" in sys.modules:
        return
    for name in ("triton", "triton.language", "triton.language.extra", "triton.language.extra.libdevice",
                 "triton.runtime", "triton.runtime.autotuner", "triton.runtime.jit", "triton.testing",
                 "triton.backends", "triton.compiler", "triton.tools", "triton.language.core"):
        m = _Mod(name)
        m.__path__ = []
        m.__version__ = "3.2.0"
        sys.modules[name] = m
    sys.modules["triton"].language = sys.modules["triton.language"]
