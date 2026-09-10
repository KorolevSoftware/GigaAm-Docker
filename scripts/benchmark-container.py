#!/usr/bin/env python3
"""Measure one image with an embedded RNNT model; requires Docker and curl."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import secrets
import statistics
import subprocess
import threading
import time
import urllib.request


def command(*args):
    return subprocess.check_output(args, text=True).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', required=True)
    parser.add_argument('--video', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--port', type=int, default=18080)
    parser.add_argument('--runs', type=int, default=3)
    args = parser.parse_args()
    if args.runs < 1:
        parser.error('--runs must be at least 1')
    args.video = args.video.resolve(strict=True)
    args.output.mkdir(parents=True, exist_ok=True)
    name = f'gigaam-benchmark-{os.getpid()}'
    key = secrets.token_hex(32)
    env = dict(os.environ, GIGAAM_API_KEY=key)
    base = f'http://127.0.0.1:{args.port}'
    samples = []
    errors = []
    phase = 'startup'
    finished = threading.Event()
    sampler = None
    result = {'image': args.image, 'video_name': args.video.name,
              'started_utc': datetime.now(timezone.utc).isoformat(),
              'video_bytes': args.video.stat().st_size, 'threads': 4,
              'model': 'gigaam-v3-e2e-rnnt', 'sample_interval_seconds': 0.5,
              'image_id': command('docker', 'image', 'inspect', args.image, '--format', '{{.Id}}'),
              'image_bytes': int(command('docker', 'image', 'inspect', args.image, '--format', '{{.Size}}')),
              'measurements': []}
    memory_script = """cat /sys/fs/cgroup/memory.current
cat /sys/fs/cgroup/memory.peak
cat /sys/fs/cgroup/memory.stat
cat /sys/fs/cgroup/cpu.stat
cat /proc/1/status"""

    def sample():
        started = time.monotonic()
        while not finished.is_set():
            try:
                lines = command('docker', 'exec', name, 'sh', '-c', memory_script).splitlines()
                fields = {}
                for line in lines[2:]:
                    parts = line.replace(':', '').split()
                    if len(parts) >= 2 and parts[1].isdigit():
                        fields[parts[0]] = int(parts[1])
                current = int(lines[0])
                samples.append({'t': time.monotonic() - started, 'phase': phase,
                                'cgroup_bytes': current, 'cgroup_peak_bytes': int(lines[1]),
                                'working_set_bytes': max(0, current - fields['inactive_file']),
                                'rss_bytes': fields['VmRSS'] * 1024,
                                'hwm_bytes': fields['VmHWM'] * 1024,
                                'cpu_usec': fields['usage_usec']})
            except Exception as error:
                errors.append(str(error))
            finished.wait(0.5)

    try:
        subprocess.run(['docker', 'run', '-d', '--name', name,
                        '-p', f'127.0.0.1:{args.port}:8080',
                        '--security-opt', 'no-new-privileges:true', '--cap-drop', 'ALL',
                        '-e', 'GIGAAM_API_KEY', '-e', 'GIGAAM_ONNX_THREADS=4',
                        '-e', 'GIGAAM_MODEL=gigaam-v3-e2e-rnnt', args.image],
                       env=env, check=True, stdout=subprocess.DEVNULL)
        sampler = threading.Thread(target=sample)
        sampler.start()
        deadline = time.monotonic() + 180
        while True:
            try:
                req = urllib.request.Request(base + '/v1/models', headers={'Authorization': f'Bearer {key}'})
                with urllib.request.urlopen(req, timeout=2) as response:
                    if response.status == 200:
                        break
            except Exception:
                if time.monotonic() > deadline:
                    raise RuntimeError('Model did not become ready within 180 seconds')
                time.sleep(0.5)
        phase = 'idle_before'
        time.sleep(12)
        result['environment'] = command('docker', 'exec', name, 'sh', '-c',
                                        'cat /etc/os-release; uname -m; ffmpeg -version | head -n 1; du -sk /models /usr/local/lib')
        result['allocator_environment'] = command('docker', 'exec', name, 'sh', '-c',
                                                'printenv LD_PRELOAD || true')
        result['allocator_mappings'] = command('docker', 'exec', name, 'sh', '-c',
                                             "grep -E 'libmimalloc|ld-musl' /proc/1/maps || true")
        for index in range(args.runs + 1):
            phase = 'cold_request' if index == 0 else f'run_{index}'
            output = args.output / f'{phase}.json'
            started = time.monotonic()
            curl = subprocess.run(['curl', '--silent', '--show-error', '--fail-with-body',
                                   '--max-time', '600', '--config', '-',
                                   '-F', f'file=@"{args.video}"',
                                   '-F', 'model=gigaam-v3-e2e-rnnt',
                                   '--output', str(output), base + '/v1/audio/transcriptions'],
                                  input=f'header = "Authorization: Bearer {key}"\n', text=True,
                                  capture_output=True)
            elapsed = time.monotonic() - started
            if curl.returncode:
                raise RuntimeError(f'curl failed: {curl.stderr}; response: {output.read_text()}')
            payload = json.loads(output.read_text())
            text = payload['text']
            (args.output / f'{phase}.txt').write_text(text + '\n')
            logs = command('docker', 'logs', name)
            rows = []
            for line in logs.splitlines():
                try:
                    row = json.loads(line)
                    if row.get('stage') == 'complete':
                        rows.append(row)
                except ValueError:
                    pass
            metrics = rows[-1]
            assert metrics['code'] == 'ok', metrics
            result['measurements'].append({'phase': phase, 'client_seconds': elapsed,
                                           'text_sha256': hashlib.sha256(text.encode()).hexdigest(),
                                           'metrics': metrics})
            print(args.image, phase, f'inference={metrics["inference_ms"]/1000:.3f}s total={elapsed:.3f}s', flush=True)
            phase = 'between_requests'
            time.sleep(2)
        phase = 'idle_after'
        time.sleep(12)
        result['container_rw_bytes'] = int(command('docker', 'inspect', '--size', name, '--format', '{{.SizeRw}}'))
        result['container_rootfs_bytes'] = int(command('docker', 'inspect', '--size', name, '--format', '{{.SizeRootFs}}'))
        for state in ['idle_before', 'idle_after']:
            selected = [s for s in samples if s['phase'] == state]
            result[state] = {k: statistics.median(s[k] for s in selected)
                             for k in ['working_set_bytes', 'rss_bytes', 'cgroup_bytes']}
        active = [s for s in samples if s['phase'] == 'cold_request' or s['phase'].startswith('run_')]
        result['processing_peak'] = {k: max(s[k] for s in active)
                                     for k in ['working_set_bytes', 'rss_bytes', 'hwm_bytes', 'cgroup_bytes']}
        result['sampling_errors'] = errors
    finally:
        finished.set()
        if sampler:
            sampler.join()
        try:
            (args.output / 'server.log').write_text(command('docker', 'logs', name))
        finally:
            (args.output / 'samples.json').write_text(json.dumps(samples, indent=2) + '\n')
            (args.output / 'results.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
            subprocess.run(['docker', 'rm', '-f', name], check=False, stdout=subprocess.DEVNULL)


if __name__ == '__main__':
    main()
