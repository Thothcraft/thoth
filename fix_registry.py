import sys
sys.path.insert(0, '/home/gad/Desktop/thoth/src')
from pathlib import Path
from backend.model_runtime import ModelRegistry

reg = ModelRegistry(Path('/home/gad/Desktop/thoth/models/user'))
keep = {'radar': None, 'fusion': None}
for m in reg.list():
    md = m.get('metadata') or {}
    if md.get('execution') != 'minute':
        reg.delete(m['id'])
        print('DELETED stale chunk model:', md.get('name'))
        continue
    inputs = md.get('inputs') or []
    kind = 'fusion' if any(i.get('sensor') == 'csi' for i in inputs) else 'radar'
    if keep[kind] is None:
        keep[kind] = m['id']
        reg.set_enabled(m['id'], True)
        print('KEPT+ENABLED', kind, ':', md.get('name'))
    else:
        reg.delete(m['id'])
        print('DELETED duplicate', kind, ':', md.get('name'))
print('final:', [(x.get('metadata') or {}).get('name') for x in reg.list()])
