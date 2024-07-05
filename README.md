# Oral Squamous Cell Carcinoma Domain Shift Classification

This study was done using Python >3.11.9, please guarantee python version before running anyy scripts in order to guarantee compatibility.

## 📥 Downloading data

The dataset used for training was [NDB-UFES: An oral cancer and leukoplakia dataset composed of histopathological images and patient data](https://data.mendeley.com/datasets/bbmmm4wgr8/4), curated by Maria Clara Falcão Ribeiro de Assis, Leandro Muniz de Lima, Liliana Aparecida Pimenta de Barros, Tânia Regina Velloso, Renato Krohling, Danielle Camisasca. Published in March 16, 2023. Mendeley Data, V4 , doi: https://doi.org/10.17632/bbmmm4wgr8.4. 

In order to use the images, the directory `data` was created at the roo of the repositoruy. The folder `images` and the file `ndb-ufes.csv` obtaind from mendeley was moved to `data`. For the patches images, the data used was obtained by Labcin instead of the format available at mendeley, this is so that the original image names can be used. The original image names contain the parent image name, having this information is vital to guarantee that no contamination is made during train/test splits. 

The data directory contained the following format:
```
data
|---- ndb-ufes
|     |---- images
|     |      |-- image.png
|     |      |-- image.png
|     |      |-- ...
|     |---- patches
|     |      |-- carcinoma
|     |      |    |-- image.png
|     |      |    |-- image.png
|     |      |    |-- ....png
|     |      |-- no_dysplasia
|     |      |    |-- image.png
|     |      |    |-- image.png
|     |      |    |-- ....png
|     |      |-- dysplasia
|     |      |    |-- image.png
|     |      |    |-- image.png
|     |      |    |-- ....png
|     |---- ndb-ufes.csv
|---- rahman
|     |---- First Set
|     |      |-- 100x Normal Oral Cavity Histopathological Images
|     |      |    |-- image.jpg
|     |      |    |-- image.jpg
|     |      |    |-- ....jpg
|     |      |-- 100x OSCC Histopathological Images
|     |      |    |-- image.jpg
|     |      |    |-- image.jpg
|     |      |    |-- ....jpg
|     |---- Second Set
|     |      |-- 400x Normal Oral Cavity Histopathological Images
|     |      |    |-- image.jpg
|     |      |    |-- image.jpg
|     |      |    |-- ....jpg
|     |      |-- 400x OSCC Histopathological Images
|     |      |    |-- image.jpg
|     |      |    |-- image.jpg
|     |      |    |-- ....jpg
```

The dataset used for test was [Histopathological imaging database for Oral Cancer analysis](https://data.mendeley.com/datasets/ftmp4cvtmb/2), curated by Tabassum Yesmin Rahman, Lipi B. Mahanta, Anup K. Das, Jagannath D. Sarma. Published in January 9, 2023. Mendeley Data, V1, doi: 10.17632/ftmp4cvtmb.2

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

PyTorch was used in this project. While it's included in the requirements, I recommend following [PyTorch's how to install guide](https://pytorch.org/), substituing with the versions included in the requirements. This way you can guarantee it's compatible with your CUDA version. For example:

```sh
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu118
```

The
## Chapters

### Research phase

1. Research about domain shift
2. Research about SOTA image classifiers
3. Research about domain shift applied to histopathology images
4. Research about OSCC image classification
5. Research about image processing applied to histopathology images

### Experimental phase

Base model: DenseNet121

Part 1 <br>

1. Split datasets in test and train for both Rahman and P-NDB-UFES.
2. Run training experiments with patches dataset with the same transformation
3. Run training experiments with Rahman dataset with the same transformation

Part 2 <br>

4. Test P-NDB-UFES model with P-NDB-UFES test set
5. Test P-NDB-UFES model with Rahman test set (no training done on Rahman here)
6. Test Rahman model with Rahman test set
7. Test Rahman model with P-NDB-UFES test set (no training done on P-NDB-UFES here)

Part 3 <br>

8. Training P-NDB-UFES model on Rahman training set (transfer learning) 100x + 400x
9. Test P-NDB-UFES-Rahman model with Rahman test
10. Test P-NDB-UFES-Rahman model with P-NDB-UFES test

Part 4 <br>

11. Analyze results


## References


