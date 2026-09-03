"""Build script for the C++/htslib BAM parser extension.

Locates htslib headers and library from:
  1. conda environment ($CONDA_PREFIX/include, $CONDA_PREFIX/lib)
  2. pysam's bundled headers (for #include <htslib/sam.h>)
  3. Standard system paths (/usr/local, etc.)

Prerequisites:
    conda install -c bioconda htslib pybind11
    # or: apt install libhts-dev && pip install pybind11
"""

import glob
import os
import platform
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
            return
        self._verify_importable()

    def _verify_importable(self):
        """A linked .so is not yet a working one.

        With `-undefined dynamic_lookup` the linker will happily emit a bundle
        whose htslib symbols were never resolved -- an architecture-mismatched
        -lhts is a warning, not an error.  The failure then surfaces as an
        ImportError at run time, where the BAM parser hides it behind its
        Python fallback and the test suite hides it behind `importorskip`.
        Load each module here so a broken build says so while the user is
        still looking at the build log.
        """
        import importlib.util
        for ext in self.extensions:
            path = self.get_ext_fullpath(ext.name)
            if not os.path.exists(path):
                continue
            mod = ext.name.rsplit(".", 1)[-1]
            spec = importlib.util.spec_from_file_location(mod, path)
            try:
                spec.loader.exec_module(importlib.util.module_from_spec(spec))
            except Exception as e:
                print(
                    f"\n*** {ext.name} linked but does not load: {e}\n"
                    f"*** This is usually an htslib built for a different "
                    f"architecture than this {_interp_arch()} interpreter.\n",
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


def _interp_arch():
    """Architecture of the running interpreter, as `file`/`lipo` spell it."""
    return platform.machine()


def _libhts_arch_ok(lib_dir):
    """Does *lib_dir* hold a libhts built for the interpreter's architecture?

    On an Apple Silicon Mac running an x86_64 conda Python, Homebrew's htslib
    is arm64.  The linker then prints `ignoring file ... found architecture
    'arm64', required architecture 'x86_64'` as a WARNING, `-undefined
    dynamic_lookup` swallows the resulting unresolved symbols, and the build
    reports success while producing a .so that raises ImportError on the first
    htslib symbol.  Every test then skips with "htslib extension not built",
    the BAM parser silently drops to the 10x-slower Python path, and
    _site_dist_core -- which has no fallback -- is simply broken.  So reject a
    mismatched libhts here rather than discover it at import time.

    Non-Darwin: `file` output is less uniform and cross-arch library dirs are
    rare, so accept and let the linker error out loudly.
    """
    if sys.platform != "darwin":
        return True
    for name in ("libhts.dylib", "libhts.a"):
        cand = os.path.join(lib_dir, name)
        if not os.path.exists(cand):
            continue
        try:
            archs = subprocess.check_output(
                ["lipo", "-archs", cand], stderr=subprocess.DEVNULL,
            ).decode().split()
        except (FileNotFoundError, subprocess.CalledProcessError):
            return True     # cannot tell; do not block the build
        return _interp_arch() in archs
    return False            # headers without a library is not a usable pair


def _pysam_htslib():
    """pysam's bundled libchtslib as a link target, or None.

    It is a Python extension module built for THIS interpreter, so its
    architecture matches by construction -- which is exactly what the
    system htslib may not.  It exports the htslib C symbols, so linking
    the .so directly by path works on both macOS and Linux.
    """
    try:
        import pysam
    except ImportError:
        return None
    pkg = os.path.dirname(pysam.__file__)
    hits = sorted(glob.glob(os.path.join(pkg, "libchtslib*.so")))
    if not hits:
        return None
    return list(pysam.get_include()), hits[0]


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
        if os.path.isfile(hts_header) and _libhts_arch_ok(lib):
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
        if include_dirs and all(_libhts_arch_ok(d) for d in library_dirs):
            return include_dirs, library_dirs
        include_dirs = [d for d in include_dirs if "pysam" in d]
        library_dirs = []
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass

    # 4. Common system paths.
    for prefix in ["/usr/local", "/usr"]:
        hdr = os.path.join(prefix, "include", "htslib", "sam.h")
        lib = os.path.join(prefix, "lib")
        if os.path.isfile(hdr) and _libhts_arch_ok(lib):
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


def _make_exts():
    """The three htslib extensions: the aggregate k/n parser, the site-level
    distance-correlation backend, and the joint multi-context read-end
    extractor (the latter two need full BAMs, so the C++ path is not optional
    in practice)."""
    try:
        include_dirs, library_dirs = _find_htslib()
        libraries, extra_objects = ["hts"], []
    except RuntimeError:
        include_dirs, library_dirs, libraries, extra_objects = [], [], [], []

    if not library_dirs:
        # No system htslib matching this interpreter's architecture.  pysam
        # ships one that matches by construction; link it directly by path.
        fallback = _pysam_htslib()
        if fallback is None:
            raise RuntimeError(
                "Cannot find an htslib matching this interpreter "
                f"({_interp_arch()}).  Install one with:\n"
                "  conda install -c bioconda htslib\n"
                "  # or: apt install libhts-dev\n"
                "  # or: pip install pysam   (its bundled libchtslib is used)"
            )
        pysam_inc, pysam_lib = fallback
        include_dirs = pysam_inc + include_dirs
        libraries, extra_objects = [], [pysam_lib]
        library_dirs = [os.path.dirname(pysam_lib)]
        print(f"*** htslib: linking pysam's bundled {os.path.basename(pysam_lib)}",
              file=sys.stderr)

    include_dirs.append(_get_pybind11_include())

    def _ext(name, source):
        return Extension(
            f"nome_hic_concordance.{name}",
            sources=[f"nome_hic_concordance/ext/{source}"],
            include_dirs=include_dirs,
            library_dirs=library_dirs,
            libraries=libraries,
            extra_objects=extra_objects,
            runtime_library_dirs=library_dirs[:],
            language="c++",
        )

    return [_ext("_bam_parser_core", "bam_parser_core.cpp"),
            _ext("_site_dist_core", "site_dist_core.cpp"),
            _ext("_joint_state_core", "joint_state_core.cpp")]


# Build the extension only when actually compiling (not for metadata queries).
ext_modules = []
_BUILD_COMMANDS = {"build", "build_ext", "install", "develop",
                   "bdist_wheel", "bdist_egg", "editable_wheel"}
if _BUILD_COMMANDS & set(sys.argv):
    try:
        ext_modules = _make_exts()
    except RuntimeError as e:
        print(f"WARNING: {e}", file=sys.stderr)
        print("The package will use the pure-Python BAM parser.\n", file=sys.stderr)


setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExt},
)
