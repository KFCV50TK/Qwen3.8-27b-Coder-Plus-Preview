"""Preserve one complete adapter atomically without touching training."""
import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

RUN = Path('/workspace/runs/qwen3.8-27b-code-mixed-distill')
DEST = Path('/workspace/runs/mixed-checkpoint-snapshots')


def sha(path):
    with path.open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--step', type=int, required=True)
    args = parser.parse_args()
    source = RUN / f'checkpoint-{args.step}'
    destination = DEST / f'checkpoint-{args.step}'
    if destination.exists():
        report = json.loads((destination/'snapshot-manifest.json').read_text())
        assert all(sha(destination/name) == digest for name, digest in report['files_sha256'].items())
        print(json.dumps(report), flush=True)
        return
    state = json.loads((source/'trainer_state.json').read_text())
    assert state['global_step'] == args.step and state['max_steps'] == 1742
    assert json.loads((RUN/'start-verification.json').read_text())['source_model'] == '/workspace/models/Qwen3.8-27B'
    required = ['adapter_config.json', 'adapter_model.safetensors', 'trainer_state.json']
    assert all((source/name).is_file() for name in required)
    DEST.mkdir(parents=True, exist_ok=True)
    assert shutil.disk_usage(DEST).free > 2 * 1024**3, 'Insufficient space; preserve training files'
    temporary = Path(tempfile.mkdtemp(prefix=f'.step-{args.step}-', dir=DEST))
    files = {}
    for name in required + ['args.json', 'README.md']:
        if not (source/name).exists():
            continue
        before = sha(source/name)
        shutil.copyfile(source/name, temporary/name)
        assert sha(temporary/name) == before
        files[name] = before
    from safetensors import safe_open
    import numpy as np
    with safe_open(str(temporary/'adapter_model.safetensors'), framework='np') as handle:
        assert len(list(handle.keys())) == 992
        for key in handle.keys():
            value = handle.get_tensor(key)
            assert np.isfinite(value).all(), key
    report = {'source': str(source), 'snapshot': str(destination), 'step': args.step,
              'source_model': '/workspace/models/Qwen3.8-27B', 'files_sha256': files,
              'tensor_count': 992, 'all_finite': True, 'model_inference_performed': False}
    (temporary/'snapshot-manifest.json').write_text(json.dumps(report, indent=2))
    temporary.rename(destination)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
