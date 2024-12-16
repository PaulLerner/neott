![logo](./viz/matos-logo.png)

# neott
Source code and data for the papers by Lerner and Yvon: 
- [Towards the Machine Translation of Scientific Neologisms](https://hal.science/hal-04835653) 
- [Unlike “Likely”, “Unlike” is Unlikely: BPE-based Segmentation hurts Morphological Derivations in LLMs](https://hal.science/hal-04831106) 

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

### evaluation

Metrics are computed after generation. To reevaluate your output or our outputs 
(provided in the same repositories as the datasets, e.g. `france_terme/coling_2025/random_icl/bloom-7b1/output.json`)
use `neott.metrics` like so:
```py
import json
from neott.metrics import compute_metrics, Preprocessor

with open("france_terme/france_terme.json","rt") as file:
    data = json.load(file)

preproc = Preprocessor("fr")

targets = [item["fr"]["text"] for item in data["test"]]

with open("france_terme/coling_2025/random_icl/bloom-7b1/output.json","rt") as file:
    outputs = [json.loads(line) for line in file.read().strip().split("\n")]

for output in outputs:
    metrics = compute_metrics(predictions=output["predictions"],syns=targets,targets=targets,preproc=preproc)
    # Numbers reported in COLING Table 1
    print(output["hyperparameters"],metrics["em"])
```

### visualization
#### freq
`python -m neott.freq data/france_terme/france_terme.json data/roots/ data/france_terme/freq_roots_fr_whole_word.json --whole_word=true --batch_size=10000`

`python -m neott.freq data/france_terme/france_terme.json /gpfsdswork/dataset/OSCAR/fr_meta/ data/france_terme/freq_oscar_fr_whole_word.json --whole_word=true --batch_size=10000 --hf=false`

#### analyze

You can reproduce all analyses using `neott.viz.analyze` 
(note metrics are not recomputed but are stored in the output, you can recompute them using `neott.metrics` as instructed above)

`python -m neott.viz.analyze data/france_terme/france_terme.json exp/prompt/test/output.json --tokenizer=bigscience/bloom-7b1 --morpher=models/morph/fr/model.bin --freq_paths=data/france_terme/freq_roots_fr_whole_word.json --freq_paths+=data/france_terme/freq_oscar_fr_whole_word.json`

obviously, all optional arguments are optional:
- freq_paths for EM wrt. term occurences (COLING fig. 4)
- morpher for morph accuracy (COLING fig. 5)
- tokenizer is used to compute fertility (COLING fig. 7)



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

To reproduce the alignment analysis of Section 4.2, use `python -m neott.viz.pvs <model_name>`

For example `python -m neott.viz.pvs bigscience/bloom-7b1` gives the following table:

| model        | filter `A-Za-z` | # pairs | negatives         | P@1 (cs) | P@1 (ci) | matches upper | matches other start |    
| ------------ | --------------- | ------- | ----------------- | -------- | -------- | ------------- | ------------------- | 
| bloom-7b1    | False           | 30,496  | 30,495 intra_pair | 67.6%    | 72.2%    | 7.0%          | -                   |     
| bloom-7b1    | False           | 30,496  | 111,326 all_intra | 65.4%    | 70.2%    | 3.7%          | -                   |     
| bloom-7b1    | False           | 30,496  | 250,679 all       | 32.6%    | 32.9%    | 0.4%          | 60.3%               |     
| bloom-7b1    | True            | 13,365  | 13,364 intra_pair | 81.2%    | 90.7%    | 4.9%          | -                   |     
| bloom-7b1    | True            | 13,365  | 111,326 all_intra | 76.3%    | 85.6%    | 1.8%          | -                   |     
| bloom-7b1    | True            | 13,365  | 250,679 all       | 38.8%    | 39.3%    | 0.1%          | 59.3%               |     

In the paper, we only reported P@1 (cs) using all_intra as negatives as we believe it is the more meaningful metric.
Indeed, using all embeddings as negatives, you see that you most often match another initial-word embedding.
As a side note, notice that the match is often the same token but written with an initial upper case, hence the "case insensitive (ci)" P@1.

filter `A-Za-z` only makes a significant difference for BLOOM which vocabulary is quite noisy.

Summing up, you can pass arguments like `python -m neott.viz.pvs bigscience/bloom-7b1 --alpha_filter=true --negatives=all_intra` 
to only get the relevant metric for BLOOM, and `--alpha_filter=false` for Croissant and Llama.

# Interactive demo

Use `neott.interact` to interactively translate source terms/generate terms given a definition.

For example: `python -m neott.interact --config exp/interact.yaml` will apply our ICL method that combines Co-hyponyms and Derivation paradigms,
using FranceTerme as ICL set. You can change the parameters as for `neott.prompt`


# citation
If you use our code or data on Machine Translation, please cite:

```bib
@inproceedings{coling2025trad,
    title = {{Towards the Machine Translation of Scientific Neologisms}},
	author={Lerner, Paul and Yvon, François},
    booktitle = "Proceedings of the 31st International Conference on Computational Linguistics",
    year = "2025",
    url={https://hal.science/hal-04835653},
    publisher = "International Committee on Computational Linguistics"
}
```

If you use our code or data on BPE, please cite:
```bib
@inproceedings{coling2025pvs,
    title = {{Unlike ``Likely'', ``Unlike'' is Unlikely: BPE-based Segmentation hurts Morphological Derivations in LLMs}},
	author={Lerner, Paul and Yvon, François},
    booktitle = "Proceedings of the 31st International Conference on Computational Linguistics",
    url={https://hal.science/hal-04831106},
    year = "2025",
    publisher = "International Committee on Computational Linguistics"
}
```

Note part of the Machine Translation was published in French. If you wish to disseminate our work in French, please cite:
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
