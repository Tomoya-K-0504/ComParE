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

```