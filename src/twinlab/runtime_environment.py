"""Read-only CPU and power diagnostics for interpreting local GPU timings."""
import json
from pathlib import Path
import statistics


def snapshot(sys_root=Path('/sys')):
    def read(path):
        try:
            return path.read_text().strip()
        except OSError:
            return None

    def number(path):
        value = read(path)
        try:
            return int(value)
        except (ValueError, TypeError):
            return None

    cpu = sys_root / 'devices/system/cpu'
    pstate = {key: number(cpu / 'intel_pstate' / key)
              for key in ('max_perf_pct', 'min_perf_pct', 'no_turbo')}
    frequencies = [number(p) for p in cpu.glob('cpu[0-9]*/cpufreq/scaling_cur_freq')]
    frequencies = [f/1000 for f in frequencies if f is not None]
    thermal = []
    for zone in sorted((sys_root / 'class/thermal').glob('thermal_zone*')):
        temperature = number(zone / 'temp')
        thermal.append({'zone': zone.name, 'type': read(zone / 'type'),
                        'celsius': temperature/1000 if temperature is not None else None,
                        'trip_millicelsius': {p.name: number(p) for p in sorted(zone.glob('trip_point_*_temp'))}})
    limits = {}
    for package in (sys_root / 'class/powercap').glob('intel-rapl:*'):
        for path in package.glob('constraint_*_power_limit_uw'):
            limits[f'{package.name}/{path.name}'] = number(path)
    warnings = []
    if pstate['max_perf_pct'] is not None and pstate['max_perf_pct'] < 50:
        warnings.append('CPU maximum performance is capped below 50%; inspect before interpreting GPU throughput.')
    if limits.get('intel-rapl:0/constraint_0_power_limit_uw') == 0:
        warnings.append('CPU package sustained power limit reads 0 W; investigate OS/firmware power management.')
    return {'intel_pstate': pstate,
            'cpu_frequency_mhz': {'min': min(frequencies), 'median': statistics.median(frequencies),
                                  'max': max(frequencies)} if frequencies else None,
            'thermal': thermal, 'power_limits_uw': limits, 'warnings': warnings}


if __name__ == '__main__':
    print(json.dumps(snapshot(), indent=2))
