![logo](./viz/matos-logo.png)

# neott
Towards Machine Translation of Scientific Neologisms

# Installation
```bash
conda create --name=neott python=3.10 
conda activate neott
conda install pytorch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 pytorch-cuda=11.8 -c pytorch -c nvidia
git clone https://github.com/PaulLerner/neott.git
pip install -e neott
```

# Experiments
## Data/preproc

`python -m neott.data.{termium|franceterme}`


`python -m neott.data.filter`



`python -m neott.data.split`

`python -m neott.tag`


## morph
for each language
### train a classifier on SIGMORPHON/MorphyNet
#### generate data from SIGMORPHON/MorphyNet
`python -m neott.morph.labels`

#### train classifier
`python -m neott.morph.classif train`

### predict on data
`python -m neott.morph.classif --model_path=models/morph/fr/model.bin --lang=fr predict data/termium_symptoms/termium_symptoms.json`

`python -m neott.morph.classif --model_path=models/morph/en/model.bin --lang=en predict data/termium_symptoms/termium_symptoms.json`



## prompt LLM
### validate prompt hyperparam
`python -m neott.prompt --config=exp/prompt/config.yaml`

## translate with mBART
TODO

## visualization
### freq
`python -m neott.freq data/termium_symptoms/termium_symptoms.json data/roots/ data/termium_symptoms/freq_roots_fr_whole_word.json --whole_word=true --batch_size=10000`

`python -m neott.freq data/termium_symptoms/termium_symptoms.json /gpfsdswork/dataset/OSCAR/fr_meta/ data/termium_symptoms/freq_oscar_fr_whole_word.json --whole_word=true --batch_size=10000 --hf=false`

