"""Build the compiled fisheye module in place.

    .venv/Scripts/python.exe src/sensors/_fisheye_cpp/build.py build_ext --inplace

Needs MSVC (Build Tools 2022 is present on this machine) and pybind11.
The resulting _fisheye*.pyd lands next to this file and `fisheye.py` imports
it from there.
"""
import sys
from pathlib import Path

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

HERE = Path(__file__).resolve().parent

extra = ["/O2", "/fp:fast", "/D_USE_MATH_DEFINES"] if sys.platform == "win32" \
    else ["-O3", "-ffast-math"]

setup(
    name="_fisheye",
    ext_modules=[Pybind11Extension("_fisheye", [str(HERE / "fisheye.cpp")],
                                   extra_compile_args=extra, cxx_std=17)],
    cmdclass={"build_ext": build_ext},
    script_args=sys.argv[1:] or ["build_ext", "--inplace"],
    options={"build_ext": {"build_lib": str(HERE)}},
)
