from pathlib import Path


dirs = ['input/processed', 'output', 'model', 'notebook', 'src/preprocess', 'log', 'report', 'tasks']
tasks = ['mask', 'elderly', 'breathing']
for task in tasks:
    for dir in dirs:
        Path(f'{task}/{dir}').mkdir(parents=True, exist_ok=True)
