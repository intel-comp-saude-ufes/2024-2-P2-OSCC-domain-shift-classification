import pathlib as pl
import os

PATCH_PATH = pl.Path(os.path.abspath("")) / pl.Path('../data') / pl.Path('patches')
IMAGE_PATH = pl.Path(os.path.abspath("")) / pl.Path('../data') / pl.Path('images') 
METADATA_PATH = pl.Path(os.path.abspath("")) / pl.Path('../data') / pl.Path('ndb-ufes.csv') 