"""Build script for the C++/htslib BAM parser extension.

Locates htslib headers and library from:
  1. conda environment ($CONDA_PREFIX/include, $CONDA_PREFIX/lib)
  2. pysam's bundled headers (for #include <htslib/sam.h>)
  3. Standard system paths (/usr/local, etc.)

Prerequisites:
    conda install -c bioconda htslib pybind11
    # or: apt install libhts-dev && pip install pybind11
"""

import os
import subprocess
import sys

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext


class BuildExt(build_ext):
    """Custom build_ext that adds C++14 flag and handles build failures gracefully."""

    def build_extensions(self):
        for ext in self.extensions:
            if self.compiler.compiler_type == "unix":
                ext.extra_compile_args.extend(["-std=c++14", "-O2"])
            elif self.compiler.compiler_type == "msvc":
                ext.extra_compile_args.append("/std:c++14")
        try:
            super().build_extensions()
        except Exception as e:
            print(
                f"\n*** C++ extension build failed: {e}\n"
                f"*** The package will still work using the pure-Python BAM parser.\n"
                f"*** To enable the fast C++ backend, install htslib:\n"
                f"***   conda install -c bioconda htslib pybind11\n",
                file=sys.stderr,
            )

    def run(self):
        try:
            super().run()
        except Exception:
            print(
                "\n*** Skipping C++ extension (build error). "
                "Pure-Python fallback will be used.\n",
                file=sys.stderr,
            )


def _find_htslib():
    """Locate htslib include and library directories.

    Returns (include_dirs, library_dirs) or raises RuntimeError.
    """
    include_dirs = []
    library_dirs = []

    # 1. Conda environment (most common for bioinformatics).
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        inc = os.path.join(conda_prefix, "include")
        lib = os.path.join(conda_prefix, "lib")
        hts_header = os.path.join(inc, "htslib", "sam.h")
        if os.path.isfile(hts_header):
            include_dirs.append(inc)
            library_dirs.append(lib)
            return include_dirs, library_dirs

    # 2. pysam's bundled headers (has headers but NOT libhts.so).
    #    We still need them for the #include paths.
    try:
        import pysam
        include_dirs.extend(pysam.get_include())
    except ImportError:
        pass

    # 3. pkg-config (works on Debian/Ubuntu with libhts-dev).
    try:
        cflags = subprocess.check_output(
            ["pkg-config", "--cflags", "htslib"], stderr=subprocess.DEVNULL,
        ).decode().strip().split()
        libs = subprocess.check_output(
            ["pkg-config", "--libs-only-L", "htslib"], stderr=subprocess.DEVNULL,
        ).decode().strip().split()
        for f in cflags:
            if f.startswith("-I"):
                include_dirs.append(f[2:])
        for f in libs:
            if f.startswith("-L"):
                library_dirs.append(f[2:])
        if include_dirs:
            return include_dirs, library_dirs
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass

    # 4. Common system paths.
    for prefix in ["/usr/local", "/usr"]:
        hdr = os.path.join(prefix, "include", "htslib", "sam.h")
        lib = os.path.join(prefix, "lib")
        if os.path.isfile(hdr):
            include_dirs.append(os.path.join(prefix, "include"))
            library_dirs.append(lib)
            return include_dirs, library_dirs

    # 5. If we only have pysam headers (no libhts), still return them
    #    and let the linker try the system default search paths.
    if include_dirs:
        return include_dirs, library_dirs

    raise RuntimeError(
        "Cannot find htslib. Install it with:\n"
        "  conda install -c bioconda htslib\n"
        "  # or: apt install libhts-dev"
    )


def _get_pybind11_include():
    try:
        import pybind11
        return pybind11.get_include()
    except ImportError:
        raise RuntimeError(
            "pybind11 is required to build the C++ extension.\n"
            "Install with: pip install pybind11  (or: conda install pybind11)"
        )


def _make_ext():
    include_dirs, library_dirs = _find_htslib()
    include_dirs.append(_get_pybind11_include())

    return Extension(
        "nome_hic_concordance._bam_parser_core",
        sources=["nome_hic_concordance/ext/bam_parser_core.cpp"],
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=["hts"],
        runtime_library_dirs=library_dirs[:],
        language="c++",
    )


# Build the extension only when actually compiling (not for metadata queries).
ext_modules = []
_BUILD_COMMANDS = {"build", "build_ext", "install", "develop",
                   "bdist_wheel", "bdist_egg", "editable_wheel"}
if _BUILD_COMMANDS & set(sys.argv):
    try:
        ext_modules = [_make_ext()]
    except RuntimeError as e:
        print(f"WARNING: {e}", file=sys.stderr)
        print("The package will use the pure-Python BAM parser.\n", file=sys.stderr)


setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExt},
)
