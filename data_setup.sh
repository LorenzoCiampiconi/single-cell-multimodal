#!/bin/sh
kaggle competitions download -c open-problems-multimodal
unzip open-problems-multimodal.zip -d data/
python single_cell_multimodal_core/data_handling.py