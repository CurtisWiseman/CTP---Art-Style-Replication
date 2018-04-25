import sys
from cx_Freeze import setup, Executable

# Dependencies are automatically detected, but it might need fine tuning.
build_exe_options = {"packages": ["os","numpy","ctypes","tensorflow","scipy","collections","operator","random","xml","keras","win32"], "excludes": ["tkinter"]}

# GUI applications require a different base on Windows (the default is for a
# console application).
base = None
if sys.platform == "win32":
    base = "console"


setup(  name = "ArtStyleReplicator",
    version = "1.0",
    description = "Genetic Adversarial Network",
    options = {"build_exe": build_exe_options},
    executables = [Executable("genegan.py", base = base)])