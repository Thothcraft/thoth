"""Optional survey metadata for room/v1. Missing values stay unknown."""
import math

def number(v):
    return isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v)

def spatial(v):
    if not isinstance(v,dict):return None
    origin=v.get('origin_enu_m'); heading=v.get('heading_deg'); floor=v.get('floor')
    valid_origin=isinstance(origin,list) and len(origin)==3 and all(number(n) for n in origin)
    valid_floor=isinstance(floor,int) and not isinstance(floor,bool)
    if v.get('surveyed') is True and not (valid_origin and number(heading) and valid_floor):
        raise ValueError('Surveyed rooms require east/north/up offsets, heading and integer floor.')
    return {'surveyed':v.get('surveyed') is True,'origin_enu_m':origin if valid_origin else None,
            'heading_deg':heading%360 if number(heading) else None,'floor':floor if valid_floor else None}

def building(v):
    if not isinstance(v,dict):return None
    anchor=v.get('anchor') or {}; lat=anchor.get('latitude');lon=anchor.get('longitude')
    if lat is not None and (not number(lat) or not -85<lat<85):raise ValueError('Latitude must be between -85 and 85 for the local map.')
    if lon is not None and (not number(lon) or not -180<=lon<=180):raise ValueError('Longitude must be between -180 and 180.')
    altitude=anchor.get('altitude_m')
    return {'id':str(v.get('id') or ''),'name':str(v.get('name') or ''),
            'anchor':{'latitude':lat,'longitude':lon,'altitude_m':altitude if number(altitude) else None}}
