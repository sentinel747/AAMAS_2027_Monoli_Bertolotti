__all__ = ["ExperimentRunner"]


def __getattr__(name: str):
    if name == "ExperimentRunner":
        from .runner import ExperimentRunner

        return ExperimentRunner
    raise AttributeError(name)
