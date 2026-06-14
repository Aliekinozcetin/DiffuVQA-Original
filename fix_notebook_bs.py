import json

with open('notebooks/run_diffuvqa_colab.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

src32 = ''.join(nb['cells'][32]['source'])

# src32 has backslash+n (\n as TWO chars: \ and n) between args.
# We need backslash+newline (\ then actual newline = line continuation).
idx = src32.index('!python sample_vqa_GPU.py')
end = src32.index('\n\ncheckpoint_file', idx)
cmd_block = src32[idx:end]

print('BEFORE:')
print(repr(cmd_block[:80]))

# Replace backslash+n with backslash+newline in the command block only
fixed_cmd = cmd_block.replace('\\\n', '\\\\\n')  # normalize first if needed
# Now: in the actual string, \n is backslash+n. Replace with \+newline:
fixed_cmd = cmd_block.replace('\\n', '\\\n')

print('AFTER:')
print(repr(fixed_cmd[:80]))

new_src32 = src32[:idx] + fixed_cmd + src32[end:]
nb['cells'][32]['source'] = [new_src32]

with open('notebooks/run_diffuvqa_colab.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print('Done.')
