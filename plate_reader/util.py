"""The few helpers reader_functions.py needs."""
import subprocess
import warnings
from functools import wraps

import matplotlib.pyplot as plt


def set_style():
    plt.rcParams.update(plt.rcParamsDefault)
    plt.rcParams.update({'figure.titlesize': 26, 'figure.titleweight': 'bold', 'axes.titlesize': 22,
                         'axes.titleweight': "normal", 'axes.labelsize': 24, 'axes.labelweight': 'normal',
                         "ytick.labelsize": 20, "xtick.labelsize": 20, 'legend.fontsize': 18})
    plt.rcParams.update({'figure.figsize': (8, 6)})
    plt.rcParams.update({'savefig.dpi': 300})


def suppress_warnings(*categories, action="ignore"):
    """
    Decorator to suppress selected warning categories inside a function.
    Example: @suppress_warnings(RuntimeWarning, UserWarning)
    """
    def deco(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            with warnings.catch_warnings():
                if categories:
                    for cat in categories:
                        warnings.simplefilter(action, category=cat)
                else:
                    warnings.simplefilter(action)
                return fn(*args, **kwargs)
        return wrapper
    return deco


def code_version(path):
    """The git commit of the code at path, marked if it has uncommitted changes."""
    def git(*args):
        return subprocess.run(["git", "-C", str(path), *args], capture_output=True, text=True, check=True).stdout.strip()
    try:
        commit = git("rev-parse", "--short", "HEAD")
        return commit + ("+uncommitted changes" if git("status", "--porcelain") else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown (not a git checkout)"
