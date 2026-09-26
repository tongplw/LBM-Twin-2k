from twinlab.runtime_environment import snapshot


def test_zero_power_limit_and_cpu_cap_are_not_treated_as_missing(tmp_path):
    values = {
        'devices/system/cpu/intel_pstate/max_perf_pct': '14',
        'devices/system/cpu/intel_pstate/no_turbo': '1',
        'devices/system/cpu/cpu0/cpufreq/scaling_cur_freq': '400000',
        'class/powercap/intel-rapl:0/constraint_0_power_limit_uw': '0',
        'class/thermal/thermal_zone0/type': 'x86_pkg_temp',
        'class/thermal/thermal_zone0/temp': '39000',
        'class/thermal/thermal_zone0/trip_point_0_temp': '-274000',
    }
    for name, value in values.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value)
    result = snapshot(tmp_path)
    assert len(result['warnings']) == 2
    assert result['cpu_frequency_mhz']['median'] == 400
    assert result['thermal'][0]['celsius'] == 39
    assert result['thermal'][0]['trip_millicelsius']['trip_point_0_temp'] == -274000
    assert result['power_limits_uw']['intel-rapl:0/constraint_0_power_limit_uw'] == 0


def test_runtime_diagnostics_allow_unavailable_sysfs(tmp_path):
    result = snapshot(tmp_path)
    assert result['cpu_frequency_mhz'] is None
    assert result['warnings'] == []
