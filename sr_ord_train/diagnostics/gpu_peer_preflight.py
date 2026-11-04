"""Two-GPU hardware-only P2P preflight; no model, CPU tensor staging, or updates."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import time
import traceback

PACKAGE = Path(__file__).resolve().parents[1]
SIZES_MIB = (64, 128, 256)


def need(condition, reason):
    if not condition:
        raise PermissionError(reason)


def command_result(arguments):
    try:
        completed = subprocess.run(arguments, capture_output=True, text=True, timeout=20)
        return {'argv': arguments, 'returncode': completed.returncode,
                'stdout': completed.stdout, 'stderr': completed.stderr}
    except Exception as error:
        return {'argv': arguments, 'returncode': None, 'stdout': '', 'stderr': str(error),
                'error_type': type(error).__name__}


def memory_state(torch):
    return [{'device': index, 'free_bytes': torch.cuda.mem_get_info(index)[0],
             'total_bytes': torch.cuda.mem_get_info(index)[1],
             'allocated_bytes': torch.cuda.memory_allocated(index),
             'reserved_bytes': torch.cuda.memory_reserved(index),
             'peak_allocated_bytes': torch.cuda.max_memory_allocated(index),
             'peak_reserved_bytes': torch.cuda.max_memory_reserved(index)} for index in (0, 1)]


def synchronize_pair(torch):
    torch.cuda.synchronize(0)
    torch.cuda.synchronize(1)


def gpu_round_trip(torch, size_mib):
    """All payload creation, copies, and equality checks remain on CUDA devices."""
    size_bytes = size_mib * 1024 ** 2
    source = torch.arange(size_bytes // 4, dtype=torch.float32, device='cuda:0')
    peer = torch.empty_like(source, device='cuda:1')
    returned = torch.empty_like(source, device='cuda:0')
    need(source.dtype == peer.dtype == returned.dtype == torch.float32, 'Non-FP32 payload')
    need(source.device.type == peer.device.type == returned.device.type == 'cuda', 'CPU payload forbidden')
    synchronize_pair(torch)
    start = time.perf_counter()
    peer.copy_(source, non_blocking=True)
    synchronize_pair(torch)
    middle = time.perf_counter()
    returned.copy_(peer, non_blocking=True)
    synchronize_pair(torch)
    end = time.perf_counter()
    exact = bool(torch.equal(source, returned))  # Only a scalar verdict returns to the host.
    forward, backward = middle-start, end-middle
    need(forward > 0 and backward > 0, 'Invalid synchronized elapsed time')
    result = {'payload_mib': size_mib, 'payload_bytes': size_bytes, 'dtype': 'float32',
              'source_device': 'cuda:0', 'peer_device': 'cuda:1', 'return_device': 'cuda:0',
              'exact_round_trip': exact, 'forward_seconds': forward, 'backward_seconds': backward,
              'round_trip_seconds': end-start, 'forward_GB_per_second': size_bytes / 1e9 / forward,
              'backward_GB_per_second': size_bytes / 1e9 / backward,
              'round_trip_GB_per_second': 2 * size_bytes / 1e9 / (end-start),
              'memory_with_live_payloads': memory_state(torch), 'cpu_payload_bytes': 0}
    del source, peer, returned
    synchronize_pair(torch)
    return result


def probe(torch, run_command=command_result, transfer=gpu_round_trip):
    result = {'status': 'BLOCKED', 'new_model_generations': 0, 'optimizer_steps': 0,
              'cpu_payload_staging': False, 'cpu_offload': False,
              'torch_version': torch.__version__, 'torch_cuda_version': torch.version.cuda,
              'hostname': socket.gethostname(), 'transfers': [], 'topology_commands': []}
    for arguments in (['nvidia-smi', 'topo', '-m'], ['nvidia-smi', 'topo', '-p2p', 'r'],
                      ['nvidia-smi', 'topo', '-p2p', 'w']):
        result['topology_commands'].append(run_command(arguments))
    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        result['reason'] = 'Exactly two visible CUDA GPUs required; no transfers attempted'
        result['cuda_visible_device_count'] = torch.cuda.device_count()
        return result
    result['cuda_visible_device_count'] = 2
    result['devices'] = []
    for index in (0, 1):
        properties = torch.cuda.get_device_properties(index)
        result['devices'].append({'device': index, 'name': properties.name,
            'compute_capability': [properties.major, properties.minor],
            'total_memory_bytes': properties.total_memory,
            'multiprocessor_count': properties.multi_processor_count,
            'uuid': str(getattr(properties, 'uuid', 'UNAVAILABLE'))})
    left, right = result['devices']
    if left['name'] != right['name'] or left['compute_capability'] != right['compute_capability'] or \
            'PRO6000' not in left['name'].replace(' ', '').upper():
        result['reason'] = 'Two matching registered PRO6000 GPUs required; no transfers attempted'
        return result
    result['peer_access'] = {'0_to_1': bool(torch.cuda.can_device_access_peer(0, 1)),
                             '1_to_0': bool(torch.cuda.can_device_access_peer(1, 0))}
    result['memory_before'] = memory_state(torch)
    if not all(result['peer_access'].values()):
        result['reason'] = 'Bidirectional CUDA peer access unavailable; CPU fallback forbidden'
        return result
    if any(item['returncode'] != 0 for item in result['topology_commands']):
        result['reason'] = 'Required topology/P2P query failed; no transfers attempted'
        return result
    try:
        for index in (0, 1):
            torch.cuda.reset_peak_memory_stats(index)
        for size in SIZES_MIB:
            item = transfer(torch, size)
            result['transfers'].append(item)
            need(item['exact_round_trip'], 'GPU round-trip values changed')
        result['memory_after'] = memory_state(torch)
        result['status'] = 'GPU_PEER_HARDWARE_PASS_NOT_MODEL_EQUIVALENCE'
    except Exception as error:
        result.update(status='FAILED', error_type=type(error).__name__, error=str(error),
                      traceback=traceback.format_exc())
        result['memory_after'] = memory_state(torch)
    return result


def require_output(output, job_id):
    need(job_id.isdigit(), 'Numeric Slurm job identity required')
    output = Path(os.path.abspath(output))
    expected = PACKAGE/'run'/('gpu_peer_'+job_id)
    need(output == expected and output.parent.resolve(strict=True) == output.parent and
         not output.exists() and not output.is_symlink(),
         'Output must be a new package/run/gpu_peer_<job> directory')
    output.mkdir()
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    job_id = os.environ.get('SLURM_JOB_ID', '')
    output = require_output(args.output, job_id)
    environment = {key: os.environ.get(key) for key in ('SLURM_JOB_ID', 'SLURM_STEP_ID',
        'SLURM_NTASKS', 'SLURM_NNODES', 'SLURM_CPUS_ON_NODE', 'SLURM_CPUS_PER_TASK',
        'SLURM_MEM_PER_NODE', 'SLURM_JOB_GPUS', 'SLURM_STEP_GPUS', 'CUDA_VISIBLE_DEVICES',
        'NVIDIA_VISIBLE_DEVICES', 'CUBLAS_WORKSPACE_CONFIG', 'PYTORCH_ALLOC_CONF', 'PYTORCH_CUDA_ALLOC_CONF')}
    try:
        need(os.environ.get('SLURM_NTASKS') == '1' and os.environ.get('SLURM_NNODES') == '1',
             'Single-node single-task two-GPU allocation required')
        import torch
        result = probe(torch)
    except Exception as error:
        result = {'status': 'FAILED', 'error_type': type(error).__name__, 'error': str(error),
                  'traceback': traceback.format_exc(), 'new_model_generations': 0, 'optimizer_steps': 0,
                  'cpu_payload_staging': False, 'cpu_offload': False, 'transfers': []}
    result['environment'] = environment
    result['interpretation'] = 'Hardware access/bandwidth only; does not authorize a model split or training recovery'
    with (output/'RESULT.json').open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({'status': result['status'], 'output': str(output/'RESULT.json')}))


if __name__ == '__main__':
    main()
