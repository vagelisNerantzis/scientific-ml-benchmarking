"""YAML paths resolve relative to the configuration file, with env expansion."""
from pathlib import Path
import os,re
import yaml

def load_config(path):
    path=Path(path).resolve()
    result=yaml.safe_load(path.read_text(encoding='utf-8'))
    if not isinstance(result,dict):
        raise ValueError('Expected YAML mapping')
    result['_base']=path.parent
    return result

def resolve_path(config,key):
    value=os.path.expandvars(str(config[key]))
    if re.search(r'\$\{[^}]+\}|\$[A-Za-z_][A-Za-z0-9_]*|%[^%]+%',value):
        raise ValueError('Unresolved environment variable for '+key)
    path=Path(value).expanduser()
    return path if path.is_absolute() else config['_base']/path
