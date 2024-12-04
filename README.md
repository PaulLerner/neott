![logo](./viz/matos-logo.png)

# neott
Source code and data for the papers by Lerner and Yvon: 
- Towards the Machine Translation of Scientific Neologisms 
- Unlike “Likely”, “Unlike” is Unlikely: BPE-based Segmentation hurts Morphological Derivations in LLMs 

(Note some work was also [published in French](https://inria.hal.science/hal-04623021/))


Work done within the [MaTOS](https://anr-matos.github.io/) ANR project.

# Installation

First install pytorch via [mamba](https://github.com/mamba-org/mamba) then use pip

```bash
mamba create --name=neott python=3.10 
mamba activate neott
mamba install pytorch=2.4.1 torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
git clone https://github.com/PaulLerner/neott.git
pip install -e neott
```

# Experiments
## Translation Experiments
This section describe how to reproduce the experiments of the paper *Towards the Machine Translation of Scientific Neologisms*.

[Jump here for experiments of *Unlike “Likely”, “Unlike” is Unlikely: BPE-based Segmentation hurts Morphological Derivations in LLMs*](#BPE-Experiments)

### Download Data

- https://github.com/ANR-MaTOS/france_terme FranceTerme
- https://github.com/ANR-MaTOS/termium complete TERMIUM thesaurus (not Symptoms subset used in the TALN 2024 paper)
- https://github.com/ANR-MaTOS/symptoms TERMIUM Symptoms subset (not used in the COLING paper, only in the French TALN 2024 paper)

### prompt LLM

With random ICL:
`python -m neott.prompt --config=exp/trad/prompt/test.yaml`

Change dataset with `--eval_path=data/termium/termium.json` for example.

Note you can validate which prompt to use using:
`python -m neott.prompt --config=exp/trad/prompt/template_form.yaml`

You can evaluate a different model by using `--model_kwargs.pretrained_model_name_or_path=croissantllm/CroissantLLMBase` for example.

You can use different ICL methods:
- Domain:
    ```yaml
    selector_kwargs:
    - n_icl: 5
      selector: domain
      domain_key: "Dom"
    ```
- Co-hyponyms
    ```yaml
    selector_kwargs:
    - definition: true
      n_icl: 5
      selector: longest
    ```
- Derivation paradigms:
  - Matching the beginning of strings:
    ```yaml
    selector_kwargs:
    - definition: false
      n_icl: 5
      selector: longest
    ```
  - Matching the end of strings:
    ```yaml
    selector_kwargs:
    - definition: false
      n_icl: 5
      selector: longest
      start: false
    ```
- Combine Co-hyponyms and Derivation paradigms:
  ```yaml
    - definition: true
      n_icl: 1
      selector: longest
    - definition: false
      n_icl: 3
      selector: longest
    - definition: false
      n_icl: 1
      selector: longest
      start: false
  ```

### translate with mBART
TODO

### visualization
#### freq
`python -m neott.freq data/france_terme/france_terme.json data/roots/ data/france_terme/freq_roots_fr_whole_word.json --whole_word=true --batch_size=10000`

`python -m neott.freq data/france_terme/france_terme.json /gpfsdswork/dataset/OSCAR/fr_meta/ data/france_terme/freq_oscar_fr_whole_word.json --whole_word=true --batch_size=10000 --hf=false`

#### analyze

You can reproduce all analyses using `neott.viz.analyze` 
(note metrics are not recomputed but are stored in the output, you can recompute them using `neott.metrics`)

`python -m neott.viz.analyze data/france_terme/france_terme.json exp/prompt/test/output.json --tokenizer=bigscience/bloom-7b1 --morpher=models/morph/fr/model.bin --freq_paths=data/france_terme/freq_roots_fr_whole_word.json --freq_paths+=data/france_terme/freq_oscar_fr_whole_word.json`

obviously, all optional arguments are optional:
- freq_paths for EM wrt. term occurences (COLING fig. 4)
- morpher for morph accuracy (COLING fig. 5)
- tokenizer is used to compute fertility (COLING fig. 7)

If you do not rerun the experiments, you can use our outputs provided in the same repositories as the datasets (e.g. `france_terme/taln_2024/bloom-7b1/output.json`)



### morph
for each language

#### download data
- https://github.com/kbatsuren/MorphyNet/tree/378144f64df58c78db5245af19d16a511ccecf3a MorphyNet
- https://github.com/sigmorphon/2022SegmentationST/tree/ac161e1107e423577e922b05f8c43c6ebad6722a SIGMORPHON

#### train a classifier on SIGMORPHON/MorphyNet
##### generate data from SIGMORPHON/MorphyNet
`python -m neott.morph.labels`

##### train classifier
`python -m neott.morph.classif train`

#### predict on data
`python -m neott.morph.classif --model_path=models/morph/fr/model.bin --lang=fr predict data/france_terme/france_terme.json`

`python -m neott.morph.classif --model_path=models/morph/en/model.bin --lang=en predict data/france_terme/france_terme.json`

### Data/preproc
The datasets provided through separate repositories above have been preprocessed with the following pipeline (no need to rerun).

`python -m neott.data.{termium|franceterme}`

`python -m neott.data.filter`

`python -m neott.data.split`

`python -m neott.tag`

## BPE Experiments
This section describe how to reproduce the experiments of the paper 
*Unlike “Likely”, “Unlike” is Unlikely: BPE-based Segmentation hurts Morphological Derivations in LLMs*

### Download data
Data is in a separate repository: https://github.com/PaulLerner/unlikely

Each dataset is in a JSON file named like in the paper (e.g. attested French adjectival bases = `adj_fr.json`).

The `*_with_space.json` files correspond to the morphological segmentation experiment. 
It is the same derivative as in the standard file but with a space around the affix to enforce morphological segmentation.

For example:
```
$ python -m json.tool adj_en.json | head
{
    "train": [
        {
            "id": "unritual",
            "en": {
                "text": "unritual",
                "def": {
                    "text": "Not ritual"
                },
                
$ python -m json.tool adj_en_with_space.json | head
{
    "train": [
        {
            "id": "un ritual",
            "en": {
                "text": "un ritual",
                "def": {
                    "text": "Not ritual"
                },

```

### Prompt LLM

The main experiment (Figure 2) can be reproduced with the same code as for the translation experiments:

For random ICL:
`python -m neott.prompt --config=exp/bpe/test.yaml`

Beware that 
```yaml
  tgt: fr
  template_lang: fr
  def_lang: fr
```

Should match the dataset language (`data/adj_fr.json` in the example)

For morphological ICL (the ICL examples match the morphology of the input), use 

```yaml
selector_kwargs:
  n_icl: 5
  selector: morph
  morph_lang: fr
  morph_key: morph_label
```

For example: `python -m neott.prompt --config=exp/bpe/morph/test.yaml`

### Alignment analysis

TODO 

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
