"""Explicit cohort and time-window metrics; unfinished trips remain visible."""
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import pandas as pd

from .config import STAGES


def summarize(output, demand_folder, config, method, seed):
    output = Path(output)
    manifest = pd.read_csv(Path(demand_folder) / 'vehicles.csv')
    trip_rows = []
    for node in ET.parse(output / 'tripinfo.xml').getroot().findall('tripinfo'):
        a = node.attrib
        trip_rows.append(dict(vehicle_id=a['id'], depart_s=float(a['depart']), arrival_s=float(a['arrival']),
                              delay_s=float(a['timeLoss']) + float(a['departDelay']),
                              waiting_s=float(a['waitingTime']), stops=float(a['waitingCount']),
                              travel_s=float(a['duration']) + float(a['departDelay'])))
    trips = manifest.merge(pd.DataFrame(trip_rows), on='vehicle_id', how='left')
    trips['complete'] = trips['arrival_s'].ge(0)
    trips.to_csv(output / 'vehicle_metrics.csv', index=False)
    series = pd.read_csv(output / 'timeseries.csv')
    summaries = []
    for stage in ('all', *STAGES):
        vehicles = trips if stage == 'all' else trips[trips.stage.eq(stage)]
        times = series if stage == 'all' else series[series.stage.eq(stage)]
        complete = vehicles[vehicles.complete]
        row = dict(method=method, seed=seed, stage=stage, scheduled_vehicles=len(vehicles),
                   completed_after_drain=len(complete), unfinished_vehicles=len(vehicles) - len(complete),
                   mean_delay_s=complete.delay_s.mean(), mean_waiting_s=complete.waiting_s.mean(),
                   mean_stops=complete.stops.mean(), mean_travel_s=complete.travel_s.mean(),
                   mean_queue_veh=times.queue_veh.mean(), max_queue_veh=int(times.queue_veh.max()),
                   max_lane_queue_m=times.max_lane_queue_m.max(),
                   throughput_vehicles=int(times.crossed_vehicles.sum()),
                   arrived_in_window=int(times.arrived_vehicles.sum()))
        summaries.append(row)
    (output / 'metrics.json').write_text(json.dumps(summaries, indent=2), encoding='utf-8')
    return summaries


def aggregate(results_dir, rows):
    path = Path(results_dir)
    frame = pd.DataFrame(rows)
    frame.to_csv(path / 'metrics_by_seed.csv', index=False)
    metrics = ['mean_delay_s', 'mean_queue_veh', 'max_queue_veh', 'mean_stops', 'mean_travel_s',
               'throughput_vehicles', 'max_lane_queue_m', 'unfinished_vehicles']
    summary = frame.groupby(['method', 'stage'])[metrics].agg(['mean', 'std', 'count'])
    summary.columns = ['_'.join(c) for c in summary.columns]
    summary.reset_index().to_csv(path / 'summary.csv', index=False)
    # Paired seeds, descriptive comparisons only. Three seeds do not establish significance.
    comparisons = []
    for variant in ('dqn_full', 'dqn_near'):
        for baseline in ('fixed', 'actuated'):
            for stage in ('all', *STAGES):
                a = frame[(frame.method == variant) & (frame.stage == stage)].set_index('seed')
                b = frame[(frame.method == baseline) & (frame.stage == stage)].set_index('seed')
                if a.empty or b.empty:
                    continue
                delta = a.mean_delay_s - b.mean_delay_s
                comparisons.append(dict(method=variant, baseline=baseline, stage=stage,
                                        paired_seed_count=len(delta), delay_difference_s=delta.mean(),
                                        delay_difference_std_s=delta.std(),
                                        delay_reduction_pct=(1 - a.mean_delay_s.mean() / b.mean_delay_s.mean()) * 100))
    pd.DataFrame(comparisons).to_csv(path / 'paired_comparisons.csv', index=False)
    return frame
