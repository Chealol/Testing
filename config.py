import os
from pathlib import Path
from dotenv import load_dotenv

# Base directory for all outputs and data. Defaults to current working directory.
BASE_DIR = Path(os.getenv("NFL_BOT_BASE_DIR", Path.cwd()))

# Where to store data outputs
DATA_OUT = Path(os.getenv("NFL_BOT_DATA_DIR", BASE_DIR / "data"))
DATA_OUT.mkdir(parents=True, exist_ok=True)

# .env path for credentials
ENV_PATH = Path(os.getenv("NFL_BOT_ENV_PATH", BASE_DIR / ".env"))
if ENV_PATH.exists():
    load_dotenv(ENV_PATH)

__all__ = ["BASE_DIR", "DATA_OUT", "ENV_PATH"]
