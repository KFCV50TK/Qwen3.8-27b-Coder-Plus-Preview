"""Download checkpoint-300 weights from GitHub Release and verify SHA256."""
import argparse
import hashlib
import json
import os
import time
import urllib.request
from pathlib import Path

REPOSITORY = 'KFCV50TK/Qwen3.8-27b-Coder-Plus-Preview'
TAG = 'checkpoint-300-preview'


def open_url(url):
    return urllib.request.urlopen(urllib.request.Request(url,
        headers={'User-Agent':'Qwen-Coder-Plus-Preview-Downloader'}), timeout=90)


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def download(url, path, expected=None):
    if expected and path.exists() and digest(path) == expected:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name+'.download')
    for attempt in range(4):
        try:
            with open_url(url) as response, part.open('wb') as output:
                while block := response.read(4*1024*1024):
                    output.write(block)
            if expected and digest(part) != expected:
                raise ValueError('SHA256 mismatch: '+path.name)
            part.replace(path)
            return
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2**attempt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('Qwen3.8-27b-Coder-Plus-Preview'))
    parser.add_argument('--adapter-only', action='store_true')
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with open_url('https://api.github.com/repos/'+REPOSITORY+'/releases/tags/'+TAG) as response:
        release = json.load(response)
    assets = {asset['name']:asset['browser_download_url'] for asset in release['assets']}
    with open_url(assets['release-manifest.json']) as response:
        manifest = json.load(response)
    for item in manifest['files']:
        is_adapter = item['path'].startswith('adapter/')
        if is_adapter != args.adapter_only:
            continue
        relative = Path(item['path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Unsafe release path')
        target = root/relative
        if target.exists() and digest(target) == item['sha256']:
            print('Verified:', relative)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        parts_dir = root/'.download-parts'
        parts_dir.mkdir(exist_ok=True)
        temporary = target.with_name(target.name+'.assembling')
        with temporary.open('wb') as output:
            for part in item['parts']:
                saved = parts_dir/part['asset']
                download(assets[part['asset']], saved, part['sha256'])
                with saved.open('rb') as stream:
                    while block := stream.read(4*1024*1024):
                        output.write(block)
                saved.unlink()
        if temporary.stat().st_size != item['bytes'] or digest(temporary) != item['sha256']:
            raise ValueError('Assembled file SHA256 mismatch: '+str(relative))
        temporary.replace(target)
        print('Downloaded and verified:', relative, flush=True)
    (root/'download-manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print('Complete:', root)


if __name__ == '__main__':
    main()
