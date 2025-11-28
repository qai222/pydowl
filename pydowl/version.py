import pathlib

VERSION_FILE = pathlib.Path(__file__).parent / "VERSION"
__version__ = VERSION_FILE.read_text().strip() if VERSION_FILE.exists() else "0.0.0"
