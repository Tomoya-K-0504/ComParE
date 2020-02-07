from pathlib import Path


dirs = ['input/processed', 'output', 'model', 'notebook', 'src/preprocess', 'log', 'report', 'tasks']
tasks = ['mask', 'elderly', 'breathing']

for dir in dirs:
    if dir == 'tasks':
        Path(dir).mkdir(parents=True, exist_ok=True)
    for task in tasks:
        Path(f'{task}/{dir}').mkdir(parents=True, exist_ok=True)
