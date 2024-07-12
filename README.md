# Medical_NLP_Case_Study

The objective of this work is to automatically encode conditions on free text taped by practionners.
It should replace vanilla behvior when searching the right codification on a search tool bar.

Our mission falls into Name Entity Recognition, a subfield of Machine Learning that predict the right class codification for each word in a text.

## Challenges

This comes with many challenges:

- Even the best possible Machine Learning is prone to inherent data-drift related errors in production
- Giving the right codification of a current taped word depends on past taped words but also on futur taped words, making pure "real time" codification untractable
- All Machine Learning model has limit in terms of text size, but strategy exists to overcome this limit while minimizing risks of long term dependency errors.

All these challenges will be taken into account while discussing possible solutions.


## Dataset


The [Quaero](https://huggingface.co/datasets/DrBenchmark/QUAERO) dataset is a French Biomedical dataset composed of train/val/test sets with 10 possible word-level codification values.

After randomly moving some validation data to train data in order to work with a 90/10 train/val distribution we got the following train/val/test distribution

|          |   train |   val |   test |
|:---------|--------:|------:|-------:|
| #samples |    1533 |   132 |    833 |


The 10 classes are not uniformly distributed, here is a table showing class distribution for each set:

| #samples with label     |   train |   val |   test |
|:-----|--------:|------:|-------:|
| LIVB |     367 |    39 |    204 |
| PROC |     683 |    60 |    381 |
| ANAT |     375 |    33 |    195 |
| DEVI |      60 |     5 |     25 |
| CHEM |     348 |    21 |    191 |
| GEOG |      62 |     8 |     47 |
| PHYS |     183 |    22 |     96 |
| PHEN |      73 |     4 |     35 |
| DISO |     897 |    78 |    513 |
| OBJC |      45 |     2 |     28 |


## Model Finetuning


Transformers encoders has taken world by storm over the last decade.
Their pre-training capability makes them capable of learning from raw data without any annotation, allowing the community to propose powerful off-the-shelf base models, sometimes requiring a great deal of computation. 
Finetuning such base model has become the de-facto way to proceed for many tasks, including NER.

We isolate 3 MIT-licensed base models of same size for the purpose of finetuning our Quaero dataset upon:

- [quinten-datalab/AliBERT-7GB](https://huggingface.co/quinten-datalab/AliBERT-7GB)
- [Dr-BERT/DrBERT-7GB](https://huggingface.co/Dr-BERT/DrBERT-7GB)
- [numind/NuNER-multilingual-v0.1](https://huggingface.co/numind/NuNER-multilingual-v0.1)


The 2 first models are **french** and **biomedical** specific (i.e. pretrained on french-only biomedical-only data).

The last one is a multilingual and non-specific model but **SOTA** on Name Entity Recognition task.

_NB: Nevertheless, the process of funetuning NER Transformers usually calls for hundreds to thousands samples per class to avoid overfitting._

We finetuned each base model using F1 metric on validation set to early stop the training resulting in the following table that shows F1 metric on test set:

|          | quinten-datalab/AliBERT-7GB | Dr-BERT/DrBERT-7GB | numind/NuNER-multilingual-v0.1 |
|:---------|----------------------------:|-------------------:|-------------------------------:|
| F1 test  |            0.56             |         0.59       |              **0.62**              |


While _numind/NuNER-multilingual-v0.1_ provides the best metric, it provides approximatively 50% more tokens than both medical specific models as a counterpart.
The following table illustrates the shift:

|     max #tokens                        |   train |   val |   test |
|:---------------------------------------|--------:|------:|-------:|
| quinten-datalab/AliBERT-7GB            |    64   |   54  |     77 |
| Dr-BERT/DrBERT-7GB                     |    61   |   46  |     75 |
| numind/NuNER-multilingual-v0.1         |    91   |   70  |    102 |


This is due to the nature of tokenization training at base model build time, that over-tokenize rare (medical) words when pretrained on generic data (_numind/NuNER-multilingual-v0.1_).
See example below when tokenizing text `leucodystrophie métachromatique`


##### numind/NuNER-multilingual-v0.1
`le | ##uco | ##dy | ##stro | ##phie | mét | ##ach | ##roma | ##tique`

##### quinten-datalab/AliBERT-7GB
`▁leuco | dystrophie | ▁méta | chromatique`


Regarding our introduction on long text challenges, medical-specific base models are good candidate to limit long-term dependency errors.

## Metrics

We present a full table that summarize overall metrics on our best model, but also class-specific metrics:

|          |        train+val/precision   |   train+val/recall        |   train+val/f1        |   train+val/number        |   test/precision        |   test/recall        |   test/f1        |   test/number        |
|:---------|-------------------------:|----------------------:|------------------:|----------------------:|------------------------:|---------------------:|-----------------:|---------------------:|
| ANAT     |                    0.866 |                 0.853 |             0.859 |              1316|                   0.371 |                0.380 |            0.376 |              695 |
| CHEM     |                    0.894 |                 0.940 |             0.917 |              1893 |                   0.598 |                0.749 |            0.665 |              942 |
| DEVI     |                    0.590 |                 0.665 |             0.625 |               173 |                   0.283 |                0.178 |            0.218 |               73 |
| DISO     |                    0.931 |                 0.932 |             0.932 |              4384 |                   0.708 |                0.717 |            0.713 |             2336 |
| GEOG     |                    0.782 |                 0.866 |             0.822 |               112 |                   0.706 |                0.741 |            0.723 |               81 |
| LIVB     |                    0.912 |                 0.918 |             0.915 |              1131 |                   0.684 |                0.662 |            0.673 |              610 |
| OBJC     |                    0.600 |                 0.124 |             0.205 |                97 |                   0.750 |                0.044 |            0.083 |               68 |
| PHEN     |                    0.667 |                 0.010 |             0.019 |               203 |                   1.000 |                0.010 |            0.020 |               99 |
| PHYS     |                    0.680 |                 0.693 |             0.686 |               625 |                   0.371 |                0.337 |            0.353 |              332 |
| PROC     |                    0.890 |                 0.918 |             0.904 |              2608 |                   0.649 |                0.577 |            0.611 |             1394 |
| overall  |                    0.888 |                 0.883 |             0.885 |             12542 |                   0.619 |                0.610 |            0.615 |             6630 |
| balanced |                    0.781 |                 0.692 |             0.688 |              1254 |                   0.612 |                0.440 |            0.444 |              663 |

Firstly, this table confirms the overfitting risk due to lack of data (comparing `train+val/f1` and `test/f1`). Few-shot learning is a common solution to overcome such situation but while strategy and library ([SetFit](https://github.com/huggingface/setfit/tree/main/notebooks)) exists for Text Classification task, nothing as matured exists for NER as today.

Secondly, table shows very bad F1 on classes `OBJC` and `PHEN` due to very bad Recall, meaning that model is failing at codifing those two classes. This can be overcome by providing greater weights on those two classes at training time to greater penalize related errors.
Here are some examples where those two classes are missed:

`text: A propos de l' évolution et de la situation épidémiologique actuelle de la lèpre à la Guadeloupe : analyse des données du fichier central du département .`

  
|    | entity_group   | word            | type         |
|---:|:---------------|:----------------|:-------------|
|  0 | DISO           | lèpre           | pred         |
|  1 | GEOG           | Guadeloupe      | pred         |
|  2 | DISO           | lèpre           | ground truth |
|  3 | GEOG           | Guadeloupe      | ground truth |
|  4 | PHEN           | **épidémiologique** | ground truth |
|  5 | PROC           | **analyse**         | ground truth |


`text: Syndrome de Reye sévère : à propos de 14 cas pris en charge dans une unité de réanimation pédiatrique pendant 11 ans .`

|    | entity_group   | word                             | type         |
|---:|:---------------|:---------------------------------|:-------------|
|  0 | DISO           | Syndrome de Reye                 | pred         |
|  1 | DISO           | Syndrome de Reye                 | ground truth |
|  2 | OBJC           | **unité de réanimation pédiatrique** | ground truth |
|  3 | PROC           | **pris en charge**                   | ground truth |


## Model generation with few shot prompting

We propose here a different approach using Transformers decoder models this time.
Those models have shown great generation capabilities thanks to their bigger size compared to Transformers encoders, here are the pros and cons using decoders for NER:

- Pros:
  - no training needed
  - provide examples at inference time to help the model
  - provide instruction on desired output at inference time
  - benefit of already encoded knowledge (at Pretraining time and Supervised Fine-Tuning time)
  - output tokens can be streamed by the model

- Cons:
  - high power needed
  - high time inference
  - output possibly not well formatted
  - hallucinations
