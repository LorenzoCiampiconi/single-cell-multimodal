apt update && apt install -y `cat system_requirements.txt`
kaggle competitions download -c open-problems-multimodal
unzip open-problems-multimodal.zip
rm open-problems-multimodal.zip
mv *.h5 data/
mv *.csv data/
python3 data_handling.py