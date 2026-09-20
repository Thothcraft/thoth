import json, glob
ps = sorted(glob.glob('/home/gad/Desktop/thoth/data/20260916_21*/manifest.json'))
for p in ps[-4:]:
    try:
        m = json.load(open(p))
    except Exception as e:
        print('===', p, 'PARSE FAIL', e); continue
    print('===', p.split('/')[-2], m.get('status'))
    for mp in m.get('model_predictions') or []:
        tl = mp.get('timeline') or []
        e = tl[-1] if tl else {}
        print(' ', mp.get('model_name'), '|', e.get('status'), e.get('class'),
              round(e.get('confidence') or 0, 3), '| excl', e.get('windows_excluded'),
              'csi', e.get('csi_windows_valid'), '| probs', e.get('window_probabilities'),
              '| gaps', e.get('window_max_gaps'))
    for err in (m.get('errors') or []):
        print('  ERR:', err)
