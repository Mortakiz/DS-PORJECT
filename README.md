## Setup:

git clone https://github.com/Mortakiz/DS-PORJECT.git

cd DS-PROJECT

# Instal dependencies

pip install transformers torch datasets pyyaml accelerate

## How to train:

# 1- CPT
run this code in a cell (Google Colab or kaggle):

python train_cpt.py

Reads data/cpt/train.txt and data/cpt/val.txt.

Saves checkpoint to ./checkpoints/cpt_final/.

Logs initial and final perplexity.

# 2- SFT

run this code in a cell (Google Colab or kaggle):

python train_sft.py

Loads ./checkpoints/cpt_final/.

Reads data/sft_pairs.json (300 pairs).

Saves checkpoint to ./checkpoints/sft_final/.

## How to score:

run this code in a cell (Google Colab or kaggle):

python score.py

Loads tests/test_questions.csv (30 questions).

Runs all 3 models (base, CPT, SFT) with greedy decoding (temperature=0, max 50 tokens).

Saves one CSV per model in ./results/.

Columns: id, question, expected_answer, model_answer.

## Where checkpoints are:

Google Drive: https://drive.google.com/drive/folders/1Je8lV9PI_QGxm817bvAMfGjlniQ5eebz?usp=sharing

