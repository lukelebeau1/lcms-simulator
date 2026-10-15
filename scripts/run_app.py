"""Start the Streamlit app on this machine, on $PORT (set by the preview launcher), default 8599.

Local mode (LCMS_SIM_LOCAL=1): the Import page may read and write data/lab/. Bound to localhost, which also avoids
the Windows Firewall prompt.
"""

import os
import sys

from streamlit.web import cli

if __name__ == "__main__":
    os.environ["LCMS_SIM_LOCAL"] = "1"
    sys.argv = ["streamlit", "run", "app/Home.py", "--server.port", os.environ.get("PORT", "8599"),
                "--server.address", "localhost"]
    sys.exit(cli.main())
