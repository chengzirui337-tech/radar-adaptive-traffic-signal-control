"""Meaningful invariants: exact safety intervals, demand seeds, radar boundary."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / '.packages'))
from radar_control.config import Config
from radar_control.environment import IntersectionEnv
from radar_control.scenario import generate_demand
from radar_control.signals import SignalMachine


class CoreTests(unittest.TestCase):
    def test_min_green_and_transition_seconds(self):
        c = Config()
        s = SignalMachine(c, ['GGrr', 'rrGG', 'GrrG', 'rGGr'])
        for _ in range(10):
            s.request(True)
            self.assertEqual(s.mode, 'green')
            s.tick()
        s.request(True)
        observed = []
        for _ in range(4):
            observed.append(s.mode)
            s.tick()
        self.assertEqual(observed, ['yellow', 'yellow', 'yellow', 'all_red'])
        self.assertEqual((s.phase, s.mode, s.elapsed), (1, 'green', 0))

    def test_max_green_is_forced(self):
        s = SignalMachine(Config(), ['Grrr', 'rGrr', 'rrGr', 'rrrG'])
        for _ in range(60):
            s.tick()
        self.assertEqual(s.mode, 'yellow')

    def test_demand_reproducibility_and_change(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as d:
            a, b, c = [Path(d) / x for x in 'abc']
            p1 = generate_demand(a, Config(), 101)
            p2 = generate_demand(b, Config(), 101)
            p3 = generate_demand(c, Config(), 102)
            self.assertEqual(p1.read_bytes(), p2.read_bytes())
            self.assertNotEqual(p1.read_bytes(), p3.read_bytes())
            vehicles = ET.parse(p1).getroot().findall('vehicle')
            self.assertTrue(900 < len(vehicles) < 1500)
            self.assertTrue(all(0 <= float(v.get('depart')) < 4500 for v in vehicles))

    def test_ablation_cannot_see_distant_vehicle(self):
        env = object.__new__(IntersectionEnv)
        env.config = Config()
        env.signal = SignalMachine(env.config, ['Grrr'] * 4)
        env.lane_index = {f'{a}_in_{i}': n * 3 + i for n, a in enumerate(('N', 'S', 'E', 'W')) for i in range(3)}
        env.observed = []
        env.observation = 'near'
        empty = env.state()
        env.observed = [dict(lane_id='N_in_1', distance_m=100, speed_mps=12, accumulated_waiting_s=0)]
        np.testing.assert_array_equal(empty, env.state())
        env.observation = 'full'
        self.assertFalse(np.array_equal(empty, env.state()))
        self.assertEqual(env.state().shape, (env.config.state_dim,))


if __name__ == '__main__':
    unittest.main()
