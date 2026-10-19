"""Merge a plain LoRA checkpoint one original safetensors shard at a time."""
import gc
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
os.environ['ASCEND_RT_VISIBLE_DEVICES'] = ''
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.nice(10)
try:
    Path('/proc/self/oom_score_adj').write_text('1000')
except OSError:
    pass
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

torch.set_num_threads(2)
BASE = Path('/workspace/models/Qwen3.8-27B')
ADAPTER = Path('/workspace/runs/mixed-checkpoint-snapshots/checkpoint-300')
FINAL = Path('/workspace/exports/Qwen3.8-27b-Coder-Plus-Preview-step300')
TEMP = FINAL.with_name(FINAL.name + '.inprogress')
RUN = Path('/workspace/runs/preview-step300-publication')
RUN.mkdir(parents=True, exist_ok=True)


def digest(path):
    with path.open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def progress(value):
    (RUN/'merge-progress.json').write_text(json.dumps(value, indent=2))
    print(json.dumps(value), flush=True)


def main():
    started = time.time()
    snapshot = json.loads((ADAPTER/'snapshot-manifest.json').read_text())
    assert snapshot['step'] == 300 and snapshot['tensor_count'] == 992 and snapshot['all_finite']
    assert all(digest(ADAPTER/name) == value for name, value in snapshot['files_sha256'].items())
    config = json.loads((ADAPTER/'adapter_config.json').read_text())
    assert config['peft_type'] == 'LORA' and not config.get('use_dora')
    assert not config.get('fan_in_fan_out') and config.get('bias') == 'none'
    assert not config.get('rank_pattern') and not config.get('alpha_pattern')
    assert not config.get('modules_to_save') and not config.get('use_rslora')
    assert not FINAL.exists() and not TEMP.exists(), 'Refusing to overwrite an existing export'
    index = json.loads((BASE/'model.safetensors.index.json').read_text())
    shards = sorted(set(index['weight_map'].values()))
    assert shutil.disk_usage('/workspace').free > sum((BASE/s).stat().st_size for s in shards) + 20 * 2**30
    adapter = load_file(str(ADAPTER/'adapter_model.safetensors'), device='cpu')
    pairs = {}
    used_keys = set()
    for key in adapter:
        if not key.endswith('.lora_A.weight'):
            continue
        b_key = key.replace('.lora_A.weight','.lora_B.weight')
        model_key = key.removeprefix('base_model.model.').replace('.lora_A.weight','.weight')
        assert model_key in index['weight_map'], model_key
        a, b = adapter[key], adapter[b_key]
        assert a.shape[0] == b.shape[1] == config['r']
        pairs[model_key] = (a, b)
        used_keys.update((key,b_key))
    assert used_keys == set(adapter), 'Unrecognized adapter tensors must not be omitted'
    TEMP.mkdir(parents=True)
    scale = config['lora_alpha'] / config['r']
    manifest = {'base':str(BASE),'adapter':str(ADAPTER),'checkpoint_step':300,
                'checkpoint_training_loss':next(row['loss'] for row in reversed(json.loads((ADAPTER/'trainer_state.json').read_text())['log_history']) if row.get('step') == 300 and 'loss' in row),
                'training_continued_during_merge':True, 'merge_device':'cpu', 'model_inference_performed':False,'adapter_sha256':digest(ADAPTER/'adapter_model.safetensors'),
                'scale':scale,'lora_pairs':len(pairs),'shards':{},'reference_checks':[],
                'changed_elements':0,'merged_weights':0}
    merged = set()
    with torch.inference_mode():
        for number, shard in enumerate(shards, 1):
            weights = {}
            with safe_open(str(BASE/shard), framework='pt', device='cpu', backend='pread') as source:
                metadata = dict(source.metadata() or {})
                metadata.update(modification='LoRA merge at checkpoint 300; see merge-manifest.json', finetuning_checkpoint='300')
                for key in source.keys():
                    original = source.get_tensor(key)
                    assert bool(torch.isfinite(original).all()), 'Nonfinite base tensor: '+key
                    if key in pairs:
                        a,b = pairs[key]
                        delta = (b.float() @ a.float()) * scale
                        assert original.shape == delta.shape
                        value = original + delta.to(original.dtype)
                        assert bool(torch.isfinite(value).all()), key
                        changed = int((value != original).sum())
                        manifest['changed_elements'] += changed
                        if len(manifest['reference_checks']) < 3:
                            from peft import LoraConfig, get_peft_model
                            module = torch.nn.Sequential(torch.nn.Linear(a.shape[1],b.shape[0],bias=False,dtype=original.dtype))
                            module[0].weight.copy_(original)
                            wrapped = get_peft_model(module,LoraConfig(r=config['r'],lora_alpha=config['lora_alpha'],target_modules=['0']))
                            layer=wrapped.base_model.model[0]
                            layer.lora_A['default'].weight.copy_(a)
                            layer.lora_B['default'].weight.copy_(b)
                            reference=wrapped.merge_and_unload(safe_merge=True)[0].weight
                            assert torch.equal(value,reference), f'PEFT merge mismatch: {key}'
                            manifest['reference_checks'].append({'key':key,'peft_exact_match':True})
                            del reference,layer,wrapped,module
                        weights[key]=value
                        merged.add(key)
                        del delta
                    else:
                        weights[key]=original
            destination=TEMP/shard
            save_file(weights,str(destination),metadata=metadata)
            with safe_open(str(destination),framework='pt',backend='pread') as check:
                assert set(check.keys())==set(weights)
                for key, tensor in weights.items():
                    assert check.get_slice(key).get_shape()==list(tensor.shape)
            manifest['shards'][shard]={'sha256':digest(destination),'bytes':destination.stat().st_size}
            manifest['merged_weights']=len(merged)
            progress({'stage':'merge','shards_done':number,'shards_total':len(shards),'merged_weights':len(merged),'elapsed_seconds':round(time.time()-started,1)})
            weights.clear()
            del weights,original
            gc.collect()
    assert merged==set(pairs) and manifest['changed_elements']>0
    for source in BASE.iterdir():
        if source.is_file() and (source.name in {'LICENSE','NOTICE','config.json','generation_config.json','model.safetensors.index.json','tokenizer_config.json','tokenizer.json','merges.txt','vocab.json','preprocessor_config.json','video_preprocessor_config.json','chat_template.jinja','special_tokens_map.json','added_tokens.json'}):
            shutil.copy2(source,TEMP/source.name)
        elif source.is_dir() and source.name=='chat_templates':
            shutil.copytree(source,TEMP/source.name)
    manifest['elapsed_seconds']=round(time.time()-started,1)
    manifest['output']=str(FINAL)
    manifest['status']='merged_and_verified'
    (TEMP/'merge-manifest.json').write_text(json.dumps(manifest,indent=2))
    TEMP.rename(FINAL)
    (RUN/'merge-result.json').write_text(json.dumps(manifest,indent=2))
    progress({'stage':'complete','output':str(FINAL),'merged_weights':len(merged),'changed_elements':manifest['changed_elements']})


if __name__=='__main__':
    main()
