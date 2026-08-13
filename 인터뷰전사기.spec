# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path
import sys

from PyInstaller.utils.hooks import collect_all

datas = []
binaries = []
hiddenimports = []
tmp_ret = collect_all('faster_whisper')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

python_root = Path(sys.base_prefix)
datas += [
    (str(python_root / 'tcl' / 'tcl8.6'), '_tcl_data'),
    (str(python_root / 'tcl' / 'tk8.6'), '_tk_data'),
]
binaries += [
    (str(python_root / 'DLLs' / '_tkinter.pyd'), '.'),
    (str(python_root / 'DLLs' / 'tcl86t.dll'), '.'),
    (str(python_root / 'DLLs' / 'tk86t.dll'), '.'),
]
hiddenimports += [
    '_tkinter',
    'tkinter',
    'tkinter.filedialog',
    'tkinter.messagebox',
    'tkinter.scrolledtext',
    'tkinter.ttk',
]


a = Analysis(
    ['interview_transcriber_gui.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=['pyi_rth_tkinter_fix.py'],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='인터뷰전사기',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
