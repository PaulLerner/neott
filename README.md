![logo](./viz/matos-logo.png)

# neott
Source code and data for the paper 
[Vers la traduction automatique des néologismes scientifiques (Towards Machine Translation of Scientific Neologisms)](https://inria.hal.science/hal-04623021/) 
by Lerner and Yvon (2024, referred to as TALN 2024 hereafter).

Work done within the [MaTOS](https://anr-matos.github.io/) ANR project.

# Installation
```bash
conda create --name=neott python=3.10 
conda activate neott
conda install pytorch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 pytorch-cuda=11.8 -c pytorch -c nvidia
git clone https://github.com/PaulLerner/neott.git
pip install -e neott
```

# Experiments
## Download Data

- https://github.com/ANR-MaTOS/france_terme FranceTerme
- https://github.com/ANR-MaTOS/termium complete TERMIUM thesaurus (not Symptoms subset used in the TALN 2024 paper)
- https://github.com/ANR-MaTOS/symptoms TERMIUM Symptoms subset

## prompt LLM
### validate prompt hyperparam (template_form, TALN 2024 fig. 2 left)


`python -m neott.prompt --config=exp/prompt/template_form.yaml`

You can evaluate a different model by using `--model_kwargs.pretrained_model_name_or_path=croissantllm/CroissantLLMBase` for example.

### test (TALN 2024 fig. 2 right)

`python -m neott.prompt --config=exp/prompt/test.yaml`

Change dataset with `--eval_path=data/termium/termium.json` for example.

## translate with mBART
TODO

## visualization
### freq
`python -m neott.freq data/france_terme/france_terme.json data/roots/ data/france_terme/freq_roots_fr_whole_word.json --whole_word=true --batch_size=10000`

`python -m neott.freq data/france_terme/france_terme.json /gpfsdswork/dataset/OSCAR/fr_meta/ data/france_terme/freq_oscar_fr_whole_word.json --whole_word=true --batch_size=10000 --hf=false`

### metrics

You can reproduce all analyses using `neott.viz.analyze` (, tokenizer is used )

`python -m neott.viz.analyze data/france_terme/france_terme.json exp/prompt/test/output.json --tokenizer=bigscience/bloom-7b1 --morpher=models/morph/fr/model.bin --freq_paths=data/france_terme/freq_roots_fr_whole_word.json --freq_paths+=data/france_terme/freq_oscar_fr_whole_word.json`

obviously, all optional arguments are optional:
- tokenizer is used to compute fertility (TALN 2024 fig. 4)
- morpher for morph accuracy (TALN 2024 fig. 3)
- freq_paths for EM wrt. term occurences (TALN 2024 fig. 5)

## Data/preproc



`python -m neott.data.{termium|franceterme}`


`python -m neott.data.filter`



`python -m neott.data.split`

`python -m neott.tag`


## morph
for each language

### download data
- https://github.com/kbatsuren/MorphyNet/tree/378144f64df58c78db5245af19d16a511ccecf3a MorphyNet
- https://github.com/sigmorphon/2022SegmentationST/tree/ac161e1107e423577e922b05f8c43c6ebad6722a SIGMORPHON

### train a classifier on SIGMORPHON/MorphyNet
#### generate data from SIGMORPHON/MorphyNet
`python -m neott.morph.labels`

#### train classifier
`python -m neott.morph.classif train`

### predict on data
`python -m neott.morph.classif --model_path=models/morph/fr/model.bin --lang=fr predict data/symptoms/symptoms.json`

`python -m neott.morph.classif --model_path=models/morph/en/model.bin --lang=en predict data/symptoms/symptoms.json`


# citation
If you use our code or data please cite

```bib
@inproceedings{lerner:hal-04623021,
  TITLE = {{Vers la traduction automatique des n{\'e}ologismes scientifiques}},
  AUTHOR = {Lerner, Paul and Yvon, Fran{\c c}ois},
  URL = {https://inria.hal.science/hal-04623021},
  BOOKTITLE = {{35{\`e}mes Journ{\'e}es d'{\'E}tudes sur la Parole (JEP 2024) 31{\`e}me Conf{\'e}rence sur le Traitement Automatique des Langues Naturelles (TALN 2024) 26{\`e}me Rencontre des {\'E}tudiants Chercheurs en Informatique pour le Traitement Automatique des Langues (RECITAL 2024)}},
  ADDRESS = {Toulouse, France},
  EDITOR = {BALAGUER and Mathieu and BENDAHMAN and Nihed and HO-DAC and Lydia-Mai and MAUCLAIR and Julie and MORENO and Jose G and PINQUIER and Julien},
  PUBLISHER = {{ATALA \& AFPC}},
  VOLUME = {1 : articles longs et prises de position},
  PAGES = {245-261},
  YEAR = {2024},
  MONTH = Jul,
  KEYWORDS = {n{\'e}ologisme ; terminologie ; morphologie ; traduction automatique},
  PDF = {https://inria.hal.science/hal-04623021/file/9096.pdf},
  HAL_ID = {hal-04623021},
  HAL_VERSION = {v1},
}
```
