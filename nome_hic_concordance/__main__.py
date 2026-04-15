"""Allow ``python -m nome_hic_concordance`` to invoke the CLI."""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
