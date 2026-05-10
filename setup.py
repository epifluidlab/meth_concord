"""Build script for the C++/htslib BAM parser extension.

Locates htslib headers and libraries via pysam (which bundles them),
so no separate htslib installation is required.
"""

import os
import sys

from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext


class BuildExt(build_ext):
    """Custom build_ext that adds pysam and pybind11 include paths."""

    def build_extensions(self):
        # Add C++11 or C++14 flag.
        for ext in self.extensions:
            if self.compiler.compiler_type == "unix":
                ext.extra_compile_args.append("-std=c++14")
                ext.extra_compile_args.append("-O2")
            elif self.compiler.compiler_type == "msvc":
                ext.extra_compile_args.append("/std:c++14")
        super().build_extensions()


def get_pysam_paths():
    """Return (include_dirs, library_dirs, libraries) from pysam's bundled htslib."""
    try:
        import pysam
    except ImportError:
        raise RuntimeError(
            "pysam must be installed before building the C++ extension. "
            "Run: pip install pysam"
        )

    include_dirs = list(pysam.get_include())
    library_dirs = []

    # pysam bundles htslib as a shared library next to its Python modules.
    pysam_dir = os.path.dirname(pysam.__file__)
    library_dirs.append(pysam_dir)

    # On some installations, the htslib shared lib is in a lib/ subdir.
    lib_subdir = os.path.join(pysam_dir, "lib")
    if os.path.isdir(lib_subdir):
        library_dirs.append(lib_subdir)

    # Also check conda environment's lib/ for system htslib.
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        conda_include = os.path.join(conda_prefix, "include")
        conda_lib = os.path.join(conda_prefix, "lib")
        if os.path.isdir(conda_include):
            include_dirs.append(conda_include)
        if os.path.isdir(conda_lib):
            library_dirs.append(conda_lib)

    return include_dirs, library_dirs


def get_pybind11_include():
    try:
        import pybind11
        return pybind11.get_include()
    except ImportError:
        raise RuntimeError(
            "pybind11 must be installed before building. "
            "Run: pip install pybind11"
        )


def make_ext():
    include_dirs, library_dirs = get_pysam_paths()
    include_dirs.append(get_pybind11_include())

    # Runtime library search path (so the extension finds libhts at runtime).
    runtime_library_dirs = library_dirs[:]

    return Extension(
        "nome_hic_concordance._bam_parser_core",
        sources=["nome_hic_concordance/ext/bam_parser_core.cpp"],
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=["hts"],
        runtime_library_dirs=runtime_library_dirs,
        language="c++",
    )


# Only add the extension if we're actually building (not just doing metadata queries).
ext_modules = []
if any(arg in sys.argv for arg in ("build", "build_ext", "install", "develop",
                                    "bdist_wheel", "bdist_egg", "editable_wheel")):
    try:
        ext_modules = [make_ext()]
    except RuntimeError as e:
        print(f"WARNING: Cannot build C++ extension: {e}", file=sys.stderr)
        print("The package will fall back to the pure-Python BAM parser.", file=sys.stderr)


setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExt},
)
