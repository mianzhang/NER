#!/bin/bash

# Data
corpus_name=msra
corpus_dir=data/${corpus_name}
dataset_dir=${corpus_dir}/dataset

# Embeddings
embeddings_dir=data/embeddings
embeddings=${embeddings_dir}/pre_trained_100dim.model
embed_dim=100

# Checkpoints
ckpt=${corpus_dir}/ckpt

do_what=$1
if [ "${do_what}" != "clean" ]; then
    embeddings=${embeddings}.clean
fi

if [ "${do_what}" == "clean" ]; then
    python -u clean.py --word2vec=${embeddings} \
        --embed_dim=${embed_dim} >> log/${corpus_name}.clean

elif [ "${do_what}" == "bieos" ]; then
    python -u bieos.py --train=${dataset_dir}/train.txt \
        --dev=${dataset_dir}/dev.txt \
        --test=${dataset_dir}/test.txt >> log/${corpus_name}.bieos

elif [ "${do_what}" == "preprocess" ]; then
    python -u preprocess.py --train=${dataset_dir}/train.txt.bieos \
        --dev=${dataset_dir}/dev.txt.bieos \
        --test=${dataset_dir}/test.txt.bieos \
        --word2vec=${embeddings} \
        --embed_dim=${embed_dim} \
        --save_data=${ckpt}/${corpus_name} \
        --lower --shuffle >> log/${corpus_name}.preprocess

elif [ "${do_what}" == "elmo" ]; then
    python -u get_elmo.py --data=${ckpt}/${corpus_name}.data.pt \
        --save_data=${ckpt}/${corpus_name}.elmo.pt\
        --batch_size=128 --mode=average >> log/${corpus_name}.elmo

elif [ "${do_what}" == "train" ]; then
    python -u train.py --corpus_name=${corpus_name} \
        --data=${ckpt}/${corpus_name}.data.pt \
        --from_begin --device=cuda:3 --epochs=800 --drop_rate=0.2 \
        --optimizer=rmsprop --batch_size=128 --learning_rate=0.001 --max_grad_value=-1 \
        --char_feature_extractor=none --inference_layer=crf \
        --embed_dim=${embed_dim} --char_embed_dim=25 \
        --word2vec=${ckpt}/${corpus_name}.word2vec >> log/${corpus_name}.train

elif [ "${do_what}" == "train_elmo" ]; then
    python -u train.py --corpus_name=${corpus_name} \
        --data=${ckpt}/${corpus_name}.data.pt \
        --elmo_data=${ckpt}/${corpus_name}.elmo.pt \
        --from_begin --device=cuda:3 --epochs=800 --drop_rate=0.2 \
        --optimizer=rmsprop --batch_size=128 --learning_rate=0.001 --max_grad_value=-1 \
        --char_feature_extractor=none --inference_layer=crf \
        --embed_dim=${embed_dim} --char_embed_dim=25 \
        --word2vec=${ckpt}/${corpus_name}.word2vec \
        --context_emb=elmo >> log/${corpus_name}.train.elmo
fi
