from distutils.core import setup
import py2exe

setup(
        options = {'py2exe': {'bundle_files': 1, 'compressed': True, 'dll_excludes': ['w9xpopen.exe']}},
        console = [{'script': 'main.py'}],
        zipfile = None,
)
