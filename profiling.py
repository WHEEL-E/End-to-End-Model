import cProfile
import io
import os
import pstats
import sys


def profile(function):
    """
    A decorator that uses cProfile to profile a function,
    and use snakeviz to visualize the results.
    """

    def inner(*args, **kwargs):
        profile = cProfile.Profile()
        profile.enable()
        retval = function(*args, **kwargs)
        profile.disable()
        profile_stats = pstats.Stats(profile, stream=io.StringIO()).sort_stats(
            "cumulative"
        )

        if not os.path.exists("profiles"):
            os.makedirs("profiles")
        profile_stats.dump_stats(
            filename=f"profiles/profile_{os.path.split(sys.argv[0])[1][:-3]}.prof"
        )

        return retval

    return inner
