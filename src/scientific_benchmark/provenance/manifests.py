"""Portable provenance; no automatic serialization of local paths or secrets."""
from pathlib import Path
import hashlib
import importlib.metadata
import platform

def sha256_file(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):
            digest.update(chunk)
    return digest.hexdigest()

def environment_manifest():
    packages={}
    for name in ['numpy','pandas','scipy','scikit-learn','torch','torchvision','timm','Pillow','PyYAML']:
        try:
            packages[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return {'python':platform.python_version(),'packages':packages}
