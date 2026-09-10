#!/usr/bin/env python3
"""Compare two images sequentially, alternating their order; Docker and macOS host."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import threading
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--baseline', required=True)
parser.add_argument('--candidate', required=True)
parser.add_argument('--wav', type=Path, required=True)
parser.add_argument('--video', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--rounds', type=int, default=3)
args = parser.parse_args()
if args.rounds < 1:
    parser.error('--rounds must be at least 1')
root = Path(__file__).resolve().parent.parent
out = args.output.resolve()
out.mkdir(parents=True, exist_ok=False)
(out / '.gitignore').write_text('*.log\nsamples.json\ncold_request.*\nrun_*\n')
images = {'baseline': args.baseline, 'candidate': args.candidate}
inputs = {'wav': args.wav.resolve(strict=True), 'video': args.video.resolve(strict=True)}
phase = 'setup'
finished = threading.Event()
monitor_errors = []

def command(*args):
    return subprocess.check_output(args, text=True).strip()

def host_snapshot():
    swap = command('sysctl', '-n', 'vm.swapusage')
    vm = command('vm_stat')
    rows = {k.strip(): int(v) for k, v in re.findall(r'^([^:\n]+):\s+(\d+)\.', vm, re.M)}
    return {'time': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'phase': phase,
            'swap_used_mib': float(re.search(r'used = ([\d.]+)M', swap).group(1)),
            'pressure_level': int(command('sysctl', '-n', 'kern.memorystatus_vm_pressure_level')),
            'swapins': rows['Swapins'], 'swapouts': rows['Swapouts'],
            'compressor_bytes': rows['Pages occupied by compressor'] * int(re.search(r'page size of (\d+) bytes', vm).group(1)),
            'load_average': list(os.getloadavg())}

def monitor():
    with (out / 'host-memory.jsonl').open('w') as file:
        while not finished.is_set():
            try:
                file.write(json.dumps(host_snapshot()) + '\n')
                file.flush()
            except Exception as error:
                monitor_errors.append(str(error))
            finished.wait(5)

def vm_snapshot():
    return command('docker', 'run', '--rm', '--entrypoint', 'sh', images['baseline'], '-c',
                   "awk '/^MemAvailable:|^SwapTotal:|^SwapFree:/ {print}' /proc/meminfo; awk '/^pswpin |^pswpout / {print}' /proc/vmstat")

schedule = []
for round_number in range(1, args.rounds + 1):
    for media in ['wav', 'video']:
        order = ['baseline', 'candidate'] if (round_number + (media == 'video')) % 2 else ['candidate', 'baseline']
        for variant in order:
            schedule.append({'round': round_number, 'media': media, 'variant': variant,
                             'directory': f'{len(schedule)+1:02d}-{media}-{variant}'})
metadata = {'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'source_commit': command('git', 'rev-parse', 'HEAD'),
            'uptime_before': command('uptime'), 'host_before': host_snapshot(),
            'docker_info': json.loads(command('docker', 'info', '--format',
                 '{"architecture":{{json .Architecture}},"cpus":{{.NCPU}},"memory_bytes":{{.MemTotal}},"kernel":{{json .KernelVersion}}}')),
            'images': {k: {'tag': v, 'id': command('docker', 'image', 'inspect', v, '--format', '{{.Id}}')} for k,v in images.items()},
            'input_sha256': {}, 'schedule': schedule, 'completed': []}
for name, path in inputs.items():
    with path.open('rb') as file:
        metadata['input_sha256'][name] = hashlib.file_digest(file, 'sha256').hexdigest()
main_was_running = 'gigaam-docker-asr-1' in command('docker', 'ps', '--format', '{{.Names}}').splitlines()
thread = threading.Thread(target=monitor)
print('OUTPUT_DIRECTORY=' + str(out), flush=True)
try:
    if main_was_running:
        subprocess.run(['docker', 'stop', 'gigaam-docker-asr-1'], check=True, stdout=subprocess.DEVNULL)
    remaining = command('docker', 'ps', '--format', '{{.Names}}')
    if remaining:
        raise RuntimeError('Other containers are running: ' + remaining)
    thread.start()
    metadata['vm_before'] = vm_snapshot()
    for step in schedule:
        phase = step['directory']
        print('BEGIN ' + phase, flush=True)
        subprocess.run(['python3', str(root/'scripts/benchmark-container.py'),
                        '--image', images[step['variant']], '--video', str(inputs[step['media']]),
                        '--runs', '1', '--output', str(out/step['directory'])], check=True)
        snapshot = host_snapshot()
        metadata['completed'].append(dict(step, host_after=snapshot, vm_after=vm_snapshot()))
        (out/'environment.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False)+'\n')
        print('END ' + phase + ' swap_mib=' + str(snapshot['swap_used_mib']) + ' pressure=' + str(snapshot['pressure_level']), flush=True)
    metadata['status'] = 'complete'
finally:
    phase = 'finished'
    finished.set()
    if thread.is_alive():
        thread.join()
    metadata['host_after'] = host_snapshot()
    metadata['monitor_errors'] = monitor_errors
    (out/'environment.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False)+'\n')
    if main_was_running:
        subprocess.run(['docker', 'start', 'gigaam-docker-asr-1'], check=True, stdout=subprocess.DEVNULL)
    print('OUTPUT_DIRECTORY=' + str(out), flush=True)
