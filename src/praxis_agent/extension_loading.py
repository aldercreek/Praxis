"""Shared, process-lifetime imports for explicitly configured extensions."""
import hashlib
import importlib.util
import inspect
import os
import sys
from contextlib import contextmanager
from threading import RLock

_IMPORT_LOCK = RLock()


@contextmanager
def extension_modules(configurations):
    added = []
    with _IMPORT_LOCK:
        def load():
            for config in configurations:
                if not config.enabled:
                    continue
                path = config.path
                label = f"extension {config.id} ({path})"
                if path.suffix.lower() != ".py" or not path.is_file():
                    raise ValueError(f"{label}: expected an existing local .py file")
                name = "_praxis_extension_" + hashlib.sha256(os.path.normcase(str(path)).encode()).hexdigest()
                module = sys.modules.get(name)
                if module is None:
                    spec = importlib.util.spec_from_file_location(name, path)
                    if spec is None or spec.loader is None:
                        raise ValueError(f"{label}: cannot create module loader")
                    module = importlib.util.module_from_spec(spec)
                    sys.modules[name] = module
                    added.append((name, module))
                    try:
                        spec.loader.exec_module(module)
                    except (Exception, SystemExit) as error:
                        hint = (f" Install missing dependencies in the Praxis service Python environment ({sys.executable})."
                                if isinstance(error, ImportError) else "")
                        raise ValueError(f"{label}: module load failed ({type(error).__name__}): {error}.{hint}") from error
                yield config, module
        try:
            yield load()
        except BaseException:
            for name, module in added:
                if sys.modules.get(name) is module:
                    del sys.modules[name]
            raise


def marked_functions(module, attribute):
    seen = set()
    for function in vars(module).values():
        if (inspect.isfunction(function) and function.__module__ == module.__name__
                and hasattr(function, attribute) and id(function) not in seen):
            seen.add(id(function))
            yield function
