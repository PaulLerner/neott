# neot
Towards Machine Translation of Scientific Neologisms

# Installation
```bash
conda create --name=neot python=3.10 
conda activate neot
conda install pytorch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 pytorch-cuda=11.8 -c pytorch -c nvidia
git clone https://github.com/ANR-MaTOS/neot.git
pip install -e neot
```

# Experiments
## Data/preproc

`python -m neot.data.{termium|franceterme}`


`python -m neot.data.filter`



`python -m neot.data.split`

`python -m neot.tag`


## morph
for each language
### train a classifier on SIGMORPHON/MorphyNet
#### generate data from SIGMORPHON/MorphyNet
`python -m neot.morph.labels`

#### train classifier
`python -m neot.morph.classif train`

### predict on data
`python -m neot.morph.classif --model_path=models/morph/fr/model.bin --lang=fr predict data/termium_symptoms/termium_symptoms.json`

`python -m neot.morph.classif --model_path=models/morph/en/model.bin --lang=en predict data/termium_symptoms/termium_symptoms.json`



## prompt LLM
### validate prompt hyperparam
TODO config with null template_form and template_lang
`python -m neot.prompt --config=exp/prompt/config.yaml`

## translate with mBART
TODO

## visualization
### freq
`python -m neot.freq data/termium_symptoms/termium_symptoms.json data/roots/ data/termium_symptoms/freq_roots_fr_whole_word.json --whole_word=true --batch_size=10000`

`python -m neot.freq data/termium_symptoms/termium_symptoms.json /gpfsdswork/dataset/OSCAR/fr_meta/ data/termium_symptoms/freq_oscar_fr_whole_word.json --whole_word=true --batch_size=10000 --hf=false`

