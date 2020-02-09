# Setting environent

```
conda create -n compare python=3.7
conda activate compare
git clone https://github.com/Tomoya-K-0504/ComParE.git
cd ComParE
pip install requirements.txt
git submodule update -i
cd ml_pkg
pip install requirements.txt
python setup.py install
cd ../
mkdir -p apex/amp
```

### For GPU users
```
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

# Unzip data
You need to change "password" position below into the password given from the organizer.
```
cd compare
unzip ComParE2020_Mask.zip -P password -d mask/
unzip ComParE2020_Elderly.zip -P password -d elderly/
unzip ComParE2020_Breathing.zip -P password -d breathing/
python compare_dirs.py
```

# Execution
```
cd compare
python mask/tasks/experiment.py
```