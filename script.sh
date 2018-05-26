sh delete_plots.sh
echo "Simulating the 2048 game..."
python 2048-RL.py
echo "Deleting old plots from google cloud..."
gsutil -m rm -r gs://yogesh307/*
echo "Uploading plots dir to google cloud..."
gsutil -m cp -r outputs/plots gs://yogesh307/
