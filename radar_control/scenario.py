"""Build a protected four-phase intersection and seeded exogenous demand."""
import csv
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET

import numpy as np

from .config import APPROACHES, DEMANDS, STAGES, TURNS


def binary(name):
    if os.environ.get('SUMO_HOME'):
        candidate = Path(os.environ['SUMO_HOME']) / 'bin' / (name + '.exe' if os.name == 'nt' else name)
        if candidate.exists():
            return str(candidate.resolve())
    spec = importlib.util.find_spec('sumo')
    if spec and spec.origin:
        base = Path(spec.origin).parent
        # A non-ASCII absolute SUMO_HOME crashes Windows SUMO 1.28 file access.
        # Relative paths are supported; callers keep the Python working directory fixed.
        os.environ.setdefault('SUMO_HOME', os.path.relpath(base))
        for folder in (base / 'bin', base):
            candidate = folder / (name + '.exe' if os.name == 'nt' else name)
            if candidate.exists():
                return str(candidate.resolve())
    found = shutil.which(name)
    if found:
        return found
    raise RuntimeError(f'{name} is missing. Install requirements.txt first.')


def write_xml(path, root):
    ET.indent(root)
    ET.ElementTree(root).write(path, encoding='utf-8', xml_declaration=True)


def build_network(folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    nodes = ET.Element('nodes')
    ET.SubElement(nodes, 'node', id='J', x='0', y='0', type='traffic_light')
    coords = {'N': (0, 320), 'S': (0, -320), 'E': (320, 0), 'W': (-320, 0)}
    edges = ET.Element('edges')
    for a, (x, y) in coords.items():
        ET.SubElement(nodes, 'node', id=a, x=str(x), y=str(y), type='priority')
        ET.SubElement(edges, 'edge', {'id': a + '_in', 'from': a, 'to': 'J',
                                     'numLanes': '3', 'speed': '13.89', 'priority': '1'})
        ET.SubElement(edges, 'edge', {'id': a + '_out', 'from': 'J', 'to': a,
                                     'numLanes': '3', 'speed': '13.89', 'priority': '1'})
    # Destination in left / straight / right order; lane 2 / 1 / 0 respectively.
    dests = {'N': ('E', 'S', 'W'), 'S': ('W', 'N', 'E'),
             'E': ('S', 'W', 'N'), 'W': ('N', 'E', 'S')}
    cons = ET.Element('connections')
    for a in APPROACHES:
        for lane, d in zip((2, 1, 0), dests[a]):
            ET.SubElement(cons, 'connection', {'from': a + '_in', 'to': d + '_out',
                                             'fromLane': str(lane), 'toLane': str(lane)})
    for name, element in [('nodes', nodes), ('edges', edges), ('connections', cons)]:
        write_xml(folder / f'{name}.xml', element)
    # SUMO's Windows file streams need relative paths under a non-ASCII workspace.
    command = [binary('netconvert'), '--node-files', 'nodes.xml',
               '--edge-files', 'edges.xml', '--connection-files', 'connections.xml',
               '--no-turnarounds', 'true', '--tls.yellow.time', '3', '--tls.allred.time', '1',
               '--xml-validation', 'never', '--output-file', 'intersection.net.xml']
    child_env = os.environ.copy()
    child_env.pop('SUMO_HOME', None)
    result = subprocess.run(command, cwd=folder, env=child_env, capture_output=True, text=True, errors='replace')
    if result.returncode:
        raise RuntimeError(result.stderr)
    net = ET.parse(folder / 'intersection.net.xml').getroot()
    links = [c.attrib for c in net.findall('connection') if c.get('tl') == 'J']
    n_links = max(int(c['linkIndex']) for c in links) + 1
    states = []
    for phase in range(4):
        chars = ['r'] * n_links
        for c in links:
            axis = 0 if c['from'][0] in 'NS' else 2
            p = axis + (1 if c['dir'] == 'l' else 0)
            if p == phase:
                chars[int(c['linkIndex'])] = 'G'
        states.append(''.join(chars))
    lanes = {lane.get('id'): float(lane.get('length'))
             for e in net.findall('edge') if e.get('id', '').endswith('_in') for lane in e.findall('lane')}
    metadata = {'green_states': states, 'lane_lengths_m': lanes, 'destinations': dests,
                'source': 'SUMO synthetic network', 'speed_limit_mps': 13.89}
    (folder / 'network.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    config = ET.Element('configuration')
    inp = ET.SubElement(config, 'input')
    ET.SubElement(inp, 'net-file', value='intersection.net.xml')
    write_xml(folder / 'intersection.sumocfg', config)
    return metadata


def generate_demand(folder, config, seed):
    """Piecewise Poisson arrivals, not deterministic headways or imported radar."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    arrivals, demand_rows = [], []
    dests = {'N': ('E', 'S', 'W'), 'S': ('W', 'N', 'E'),
             'E': ('S', 'W', 'N'), 'W': ('N', 'E', 'S')}
    direction_probs = (.2, .2, .3, .3)
    for begin in range(0, config.horizon, config.demand_update_seconds):
        end = min(begin + config.demand_update_seconds, config.horizon)
        stage = min(begin // config.stage_seconds, 4)
        fluct = rng.uniform(-config.fluctuation, config.fluctuation)
        rate = DEMANDS[stage] * (1 + fluct)
        turns = np.maximum(np.array(TURNS[stage]) + rng.uniform(-config.turning_jitter, config.turning_jitter, 3), .001)
        turns /= turns.sum()
        count = int(rng.poisson(rate * (end - begin) / 3600))
        times = np.sort(rng.uniform(begin, end, count))
        demand_rows.append(dict(begin_s=begin, end_s=end, stage=STAGES[stage], base_veh_h=DEMANDS[stage],
                                demand_veh_h=rate, generated_vehicles=count,
                                left_ratio=turns[0], straight_ratio=turns[1], right_ratio=turns[2]))
        for t in times:
            a = str(rng.choice(APPROACHES, p=direction_probs))
            turn = int(rng.choice(3, p=turns))
            arrivals.append((float(t), a, turn))
    arrivals.sort()
    root = ET.Element('routes')
    ET.SubElement(root, 'vType', id='car', accel='2.6', decel='4.5', sigma='.5', length='5',
                  minGap='2.5', maxSpeed='13.89', speedFactor='1', speedDev='0', tau='1')
    for a in APPROACHES:
        for turn in range(3):
            ET.SubElement(root, 'route', id=f'{a}_{turn}', edges=f'{a}_in {dests[a][turn]}_out')
    manifest = []
    for i, (t, a, turn) in enumerate(arrivals):
        vid = f'v{i:06d}'
        ET.SubElement(root, 'vehicle', id=vid, type='car', route=f'{a}_{turn}',
                      depart=f'{t:.3f}', departLane=str((2, 1, 0)[turn]), departSpeed='max')
        manifest.append(dict(vehicle_id=vid, scheduled_depart_s=t, approach=a,
                             turn=('left', 'straight', 'right')[turn], stage=STAGES[min(int(t // config.stage_seconds), 4)]))
    write_xml(folder / 'demand.rou.xml', root)
    for name, rows in [('demand.csv', demand_rows), ('vehicles.csv', manifest)]:
        with (folder / name).open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (folder / 'provenance.json').write_text(json.dumps({
        'type': 'synthetic_SUMO_input', 'seed': seed, 'vehicle_count': len(manifest),
        'arrival_model': 'piecewise homogeneous Poisson; independent uniform times conditional on count',
        'config': config.to_dict()}, indent=2), encoding='utf-8')
    return folder / 'demand.rou.xml'
