import json, sys
p = sys.argv[1]
m = json.load(open(p))
print('===', p.split('/')[-2], m.get('status'))
for mp in m.get('model_predictions') or []:
    tl = mp.get('timeline') or []
    e = tl[-1] if tl else {}
    sc = e.get('scores') or {}
    print(f"  {mp.get('model_name'):18} {e.get('status'):8} {str(e.get('class')):9} empty={sc.get('empty',0):.3f} occ={sc.get('occupied',0):.3f} excl={e.get('windows_excluded')} csi={e.get('csi_windows_valid')}")
