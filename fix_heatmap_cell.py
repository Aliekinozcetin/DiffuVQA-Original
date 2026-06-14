import json, math

with open('notebooks/run_diffuvqa_colab.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

src = ''.join(nb['cells'][35]['source'])

old = 'model, _ = create_model_and_diffusion(args=args)'

new = (
    '# Derive image_resolution from checkpoint to handle old (384) vs new (224) checkpoints.\n'
    'state_dict_peek = torch.load(ckpt, map_location="cpu")\n'
    '_pos_key = "fuse.vision_encoder.visual.positional_embedding"\n'
    'if _pos_key in state_dict_peek:\n'
    '    _n_patches = state_dict_peek[_pos_key].shape[0]\n'
    '    _derived_res = int(math.sqrt(_n_patches - 1)) * 32\n'
    '    if args.image_resolution != _derived_res:\n'
    '        print(f"image_resolution override: {args.image_resolution} -> {_derived_res} (from checkpoint)")\n'
    '        args.image_resolution = _derived_res\n'
    'model, _ = create_model_and_diffusion(args=args)'
)

assert old in src, 'target line not found'
new_src = src.replace(old, new)

# Also add math import at the top if not present
if 'import math' not in new_src:
    new_src = new_src.replace('import os, json, torch, numpy as np', 'import os, json, math, torch, numpy as np')

nb['cells'][35]['source'] = [new_src]

with open('notebooks/run_diffuvqa_colab.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print('done')
