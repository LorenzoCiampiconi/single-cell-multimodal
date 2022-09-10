kaggle competitions download -c open-problems-multimodal
unzip open-problems-multimodal.zip
mv *.h5 data/
mv *.csv data/
python3 data_handling.py