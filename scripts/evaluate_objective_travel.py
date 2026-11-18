"""Compare AGX packages through act() on public 2.5D map data.

The normal policy receives simulated own movement and the exercise's known
objective. The evaluator uses exported 1 m floor/cover samples, not native
AGX collision. It measures reaching an objective and holding use for the game duration.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys
import types
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_training import complete_records


class Terrain:
    def __init__(self, data, stress_points=()):
        self.stress_points = stress_points
        grid = data['walkable_surface_candidates']
        self.origin, self.spacing = grid['origin_xy'], grid['cell_size_m']
        self.heights = defaultdict(list)
        for x, y, z in grid['cells']:
            self.heights[(x, y)].append(z)
        self.blockers = defaultdict(list)
        for wall in data['wall_cover_contours']:
            polygon = wall['polygon_xy']
            a = self.cell(min(p[0] for p in polygon), min(p[1] for p in polygon))
            b = self.cell(max(p[0] for p in polygon), max(p[1] for p in polygon))
            for x in range(a[0], b[0]+1):
                for y in range(a[1], b[1]+1):
                    self.blockers[(x, y)].append((polygon, wall['height_range_m']))

    def cell(self, x, y):
        return (math.floor((x-self.origin[0])/self.spacing),
                math.floor((y-self.origin[1])/self.spacing))

    @staticmethod
    def inside(x, y, polygon):
        odd = False
        a = polygon[-1]
        for b in polygon:
            if (a[1] > y) != (b[1] > y) and x < a[0]+(y-a[1])*(b[0]-a[0])/(b[1]-a[1]):
                odd = not odd
            a = b
        return odd

    def step(self, x, y, feet, airborne_z, before=None):
        cell = self.cell(x, y)
        # The exported grid treats these measured obstructions as ordinary
        # ~0.20 m risers. Bracket the original model only in the affected
        # neighborhoods to avoid creating new walls throughout Mirage.
        near_stall = any(abs(feet-z) < .7 and math.hypot(x-sx,y-sy) < 2.
                         for sx,sy,z in self.stress_points)
        step_height = .16 if near_stall else .48
        ceiling = feet + step_height
        if airborne_z is not None:
            ceiling = max(ceiling, airborne_z)
        candidates = [h for h in self.heights[cell] if feet-1.4 <= h <= ceiling]
        if not candidates:
            return None
        z = min(candidates, key=lambda h: abs(h-feet))
        body_z = max(z, airborne_z) if airborne_z is not None else z
        for polygon, (low, high) in self.blockers[cell]:
            if high > body_z+.48 and low < body_z+1.4 and self.inside(x, y, polygon):
                return None
        return z


class WitnessedTerrain:
    """Resolve coarse-floor false negatives near actual held-out traversals."""

    def __init__(self, base, training, mapping):
        self.base = base
        self.by_cell = defaultdict(set)
        self.map_geometry = None
        self.integrity = {}
        self.accepted = 0
        previous = None
        for row in complete_records(training):
            if row.get('type') == 'metadata':
                self.integrity['job_id'] = row['job_id']
            elif row.get('type') == 'export_complete':
                self.integrity.update({key: row[key] for key in ('rows', 'sha256')})
            o = row.get('observation')
            if o is None:
                continue
            if 'map' in o:
                self.map_geometry = o['map']['geometry_hash']
            m = o['me']
            if o['phase'] != 'live' or not m['alive'] or not m['grounded']:
                previous = None
                continue
            pos = tuple(round(m[k], 2) for k in ('x', 'y', 'z'))
            if previous and o['round_index'] == previous[0] and o['match_tick']-previous[1] == 3:
                a = previous[2]
                if .04 <= math.dist(a[:2], pos[:2]) <= .65 and abs(a[2]-pos[2]) <= .35:
                    # Keep the actually measured segment and direction. Separate
                    # nearby floors and points outside this narrow tube remain blocked.
                    self.by_cell[math.floor(a[0]), math.floor(a[1])].add((a, pos))
            previous = o['round_index'], o['match_tick'], pos
        if self.map_geometry != mapping['geometry_hash']:
            raise ValueError('Trace geometry differs from the rollout map')

    inside = staticmethod(Terrain.inside)

    def step(self, x, y, feet, airborne_z, before=None):
        z = self.base.step(x, y, feet, airborne_z)
        if z is not None or before is None or airborne_z is not None:
            return z
        dx, dy = x-before[0], y-before[1]
        key = math.floor(x), math.floor(y)
        best = None
        for ix in range(key[0]-1, key[0]+2):
            for iy in range(key[1]-1, key[1]+2):
                for a, b in self.by_cell.get((ix, iy), ()):
                    u, v = b[0]-a[0], b[1]-a[1]
                    if u*dx+v*dy < .707*math.hypot(dx, dy)*math.hypot(u, v) or abs(feet-a[2]) > .35:
                        continue
                    before_along = max(0., min(1., ((before[0]-a[0])*u+(before[1]-a[1])*v)/(u*u+v*v)))
                    if math.hypot(before[0]-a[0]-before_along*u, before[1]-a[1]-before_along*v) > .35:
                        continue
                    along = max(0., min(1., ((x-a[0])*u+(y-a[1])*v)/(u*u+v*v)))
                    offset = math.hypot(x-a[0]-along*u, y-a[1]-along*v)
                    if offset <= .35 and (best is None or offset < best[0]):
                        best = offset, a[2]+along*(b[2]-a[2])
        if best is not None:
            self.accepted += 1
            return best[1]
        return None


def load_source(path):
    if path.suffix == '.zip':
        with zipfile.ZipFile(path) as archive:
            source = archive.read('agent.py')
    else:
        source = path.read_bytes()
    module = types.ModuleType('objective_policy')
    exec(compile(source, str(path), 'exec'), module.__dict__)
    return module, hashlib.sha256(source).hexdigest()


def rollout(module, mapping, terrain, start, site, side, template, limit):
    m = dict(template['me'], **dict(zip(('x', 'y', 'z'), start['position'])),
             yaw=start['yaw'], pitch=0., vx=0., vy=0., vz=0., grounded=True,
             carrying_bomb=side=='attack', alive=True, blind_ticks=0, crouch=False, walk=False)
    o = dict(template, me=m, map=mapping, map_id=mapping['id'], static_schema_hash=mapping['geometry_hash'],
             phase='live', side=side, seat=0 if side=='attack' else 9,
             team=0 if side=='attack' else 1, attack_team=0,
             round_index=mapping['sites'].index(site), visible=[], sounds=[], heard=[],
             teammates=[], world_items=[], messages=[], receipts=[])
    goal = site['center']
    o['bomb'] = ({'state':'held', 'carrier':o['seat'], 'position':start['position']}
                 if side=='attack' else {'state':'planted','position':goal,'site':site['id'],'fuse_end_tick':2760})
    module.init({})
    closest = math.dist(start['position'], goal)
    blocked = 0
    ground_z = m['z']
    launch_time = None
    launch_z = ground_z
    start_time = None
    held_ticks = 0
    interaction = 'plant' if side == 'attack' else 'defuse'
    duration = 180 if side == 'attack' else (300 if m.get('has_kit') else 600)
    for i in range(limit*20):
        tick = 360+3*i
        o.update(match_tick=tick, round_tick=tick, phase_remaining_ticks=6900-3*i)
        if side=='attack':
            o['bomb']['position'] = [m[k] for k in ('x','y','z')]
        a = module.act(o)
        o.pop('map', None)
        if a.get('use') == interaction and m['grounded'] and math.hypot(m['vx'], m['vy']) <= .1:
            if side == 'attack':
                zone = mapping['plant_zones'][site['id']]
                in_range = (zone['min_z'] <= m['z'] <= zone['max_z']
                            and terrain.inside(m['x'], m['y'], zone['polygon']))
            else:
                in_range = math.dist([m['x'], m['y'], m['z']], goal) < 2.8
            held_ticks = held_ticks+3 if in_range else 0
        else:
            held_ticks = 0
        progress_key = interaction+'_progress'
        if held_ticks:
            o['bomb'][progress_key] = held_ticks
            if held_ticks == 3:
                start_time = i/20
        else:
            o['bomb'].pop(progress_key, None)
            start_time = None
        if held_ticks >= duration:
            return {'interaction_s': round(start_time, 2), 'completion_s': round((i+1)/20, 2),
                    'interaction_site': site['id'], 'time_for_interaction': True,
                    'closest_m': round(closest, 2), 'stalled_s': round(blocked/20, 2)}
        f, r = a.get('move_forward', 0.), a.get('move_right', 0.)
        v = 5.56 * (.4 if a.get('walk') else 1.) * (.45 if a.get('crouch') else 1.)
        yaw = m['yaw']
        dx = v*(math.cos(yaw)*f+math.sin(yaw)*r)/20
        dy = v*(-math.sin(yaw)*f+math.cos(yaw)*r)/20
        x, y = m['x'], m['y']
        if a.get('jump') and m['grounded']:
            launch_time, launch_z = i, ground_z
        airborne_z = None
        if launch_time is not None:
            # Median of 120 observed jumps in 17eb: 0.70 s, peak 1.44 m.
            phase = (i+1-launch_time)/14
            airborne_z = launch_z+4*1.44*phase*(1-phase)
        for sx, sy in ((dx,dy),(dx,0.),(0.,dy)):
            if math.hypot(sx,sy) < 1e-6:
                continue
            z = terrain.step(x+sx, y+sy, ground_z, airborne_z, before=(x, y))
            if z is not None:
                m.update(x=x+sx,y=y+sy)
                ground_z = z
                break
        if launch_time is not None and phase >= .5 and airborne_z <= ground_z+.01:
            launch_time = None
        m['grounded'] = launch_time is None
        m['z'] = ground_z if m['grounded'] else max(ground_z,airborne_z)
        requested = math.hypot(dx,dy)
        displacement = math.hypot(m['x']-x,m['y']-y)
        blocked += int(requested>.05 and displacement<.2*requested)
        m.update(yaw=(yaw+.15*a.get('yaw_rate',0.)+math.pi)%(2*math.pi)-math.pi,
                 pitch=min(1.4,max(-1.4,m['pitch']+.1*a.get('pitch_rate',0.))),
                 vx=20*(m['x']-x),vy=20*(m['y']-y),crouch=bool(a.get('crouch')),walk=bool(a.get('walk')))
        closest = min(closest,math.dist([m['x'],m['y'],m['z']], goal))
    return {'interaction_s': start_time, 'completion_s': None, 'interaction_site': None,
            'time_for_interaction': False, 'closest_m': round(closest, 2),
            'stalled_s': round(blocked/20, 2)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--map-kit', type=Path, required=True, help='Official agx-fps-kit.zip')
    p.add_argument('--observation', type=Path, required=True, help='Personal training export for input schema')
    p.add_argument('--witness-training', type=Path,
                   help='Optional independent Mirage trace for a measured-floor sensitivity check')
    p.add_argument('--baseline', type=Path, default=ROOT/'dist/agx-fps-agent-v13.zip')
    p.add_argument('--candidate', type=Path, default=ROOT/'agent.py')
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    with args.observation.open() as stream:
        template = next(r['observation'] for r in map(json.loads,stream) if r.get('observation'))
    policies = {'v13': load_source(args.baseline), 'candidate': load_source(args.candidate)}
    cases = []
    trace_integrity = None
    maps = [('mirage', '8342'), ('dust2', '5538'), ('molo', 'dde6'),
            ('nevaio', '11dd'), ('borgo', 'eb98')]
    with zipfile.ZipFile(args.map_kit) as kit:
        for name, prefix in maps:
            raw = (ROOT / f'data/private/{prefix}-static.jsonl').read_bytes()
            descriptor = json.loads((ROOT / f'data/private/{prefix}-descriptor.json').read_text())
            expected = descriptor['streams']['public_static']['sha256']
            if hashlib.sha256(raw).hexdigest() != expected:
                raise ValueError(f'{name}: public geometry checksum mismatch')
            mapping = json.loads(raw)['map']
            terrain_data = json.loads(kit.read(f'data/fps-{name}.mapdata.json'))
            if terrain_data['map_id'] != mapping['id']:
                raise ValueError(f'{name}: kit map mismatch')
            terrains = [('public_floor', Terrain(terrain_data))]
            if name == 'mirage':
                stalls = [(39.80, 71.49, 15.53), (39.35, 65.21, 14.31), (75.41, 65.56, 15.73)]
                terrains.append(('recorded_riser_neighborhoods', Terrain(terrain_data, stalls)))
                if args.witness_training:
                    measured = WitnessedTerrain(terrains[0][1], args.witness_training, mapping)
                    trace_integrity = measured.integrity
                    terrains.append(('held_out_ground_segments', measured))
            for side in ('attack', 'defend'):
                starts = mapping['spawns']['T' if side == 'attack' else 'CT']
                limit = 115 if side == 'attack' else 40
                for site in mapping['sites']:
                    # Both sides of the spawn row are fixed before comparing versions.
                    for index in (0, len(starts)-1):
                        for model, terrain in terrains:
                            if model == 'held_out_ground_segments' and (side != 'defend' or site['id'] != 'B'):
                                continue
                            case = dict(model=model, map=name, side=side, site=site['id'],
                                        spawn=index, limit_s=limit)
                            for label, (module, _) in policies.items():
                                prior = getattr(terrain, 'accepted', None)
                                case[label] = rollout(module, mapping, terrain, starts[index],
                                                      site, side, template, limit)
                                if prior is not None:
                                    case[label]['steps_verified_by_trace'] = terrain.accepted-prior
                            cases.append(case)
    result = {
        'source_sha256': {label: pair[1] for label, pair in policies.items()},
        'map_kit_sha256': hashlib.sha256(args.map_kit.read_bytes()).hexdigest(),
        'witness_training': trace_integrity,
        'limitations': [
            'Independent runs from public spawns, with an explicitly known objective and no opponents.',
            'A horizontal point moves over public 1 m floor samples and cover outlines. Native collision, '
            'player radius and inertia are omitted. Yaw response, speed, jump height and airtime are '
            'calibrated from the 17eb own-seat trace.',
            'Default step limit is 0.48 m. The sensitivity run uses a 0.16 m limit within 2 m of three '
            'recorded stalls on Mirage, bracketing observed ~0.20 m risers.',
            'The optional Mirage B check accepts a step rejected by the coarse terrain only within '
            '0.35 m horizontally and vertically of a same-direction grounded segment from the '
            'independent trace; observations remain private to the evaluator.',
            'Stalled seconds are callbacks that travel under 20% of requested motion. Completion requires '
            'continuous use inside the public plant zone or within 2.8 m of the bomb for the game duration. '
            'Travel times and outcomes apply only to this exercise.'
        ],
        'cases': cases
    }
    encoded=json.dumps(result,indent=2)+'\n'
    if args.output:
        args.output.write_text(encoded)
    else:
        print(encoded,end='')


if __name__=='__main__':
    main()
