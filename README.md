# Oral Squamous Cell Carcinoma Domain Shift Classification

This study was done using Python >3.11.9, please guarantee python version before running anyy scripts in order to guarantee compatibility.

## 📥 Downloading data

The dataset used for training was [NDB-UFES: An oral cancer and leukoplakia dataset composed of histopathological images and patient data](https://data.mendeley.com/datasets/bbmmm4wgr8/4), curated by Maria Clara Falcão Ribeiro de Assis, Leandro Muniz de Lima, Liliana Aparecida Pimenta de Barros, Tânia Regina Velloso, Renato Krohling, Danielle Camisasca. Published in March 16, 2023. Mendeley Data, V4 , doi: https://doi.org/10.17632/bbmmm4wgr8.4.

The dataset used for test was [Histopathological imaging database for Oral Cancer analysis](https://data.mendeley.com/datasets/ftmp4cvtmb/2), curated by Tabassum Yesmin Rahman. Published in January 9, 2023. Mendeley Data, V1, doi: 10.17632/ftmp4cvtmb.2

## 🔨 Configuring environment

Before installing dependencies, create and start your virtual environment. 
```sh
python -m venv {virtual_env_name}
```

on linux run the command:
```sh
source {virtual_env_name}/bin/activate
```

on window run the command:
```sh
{virtual_env_name}\Scripts\activate
```

Before installing dependencies via terminal, guarantee that the tag `{virtual_env_name}` appears before directory path. Once confirmed, run the command:

```sh
pip install -r requirements.txt
```

## Chapters

### Research phase

1. Research about domain shift
2. Research about SOTA image classifiers
3. Research about domain shift applied to histopathology images
4. Research about OSCC image classification
5. Research about image processing applied to histopathology images

### Experimental phase

1. Split datasets in test and train
2. Run training experiments with patches dataset and images in their original size (don't mix datasets)
3. Test models in all test sets
4. Study image processing transformation on Rahman's training image set
5. Apply image transformation to NDB-UFES training image set and rerun experiments
6. Test models and validate if transformations improved classification in domain shift

## References

