# CoRelation
The evaluation implementation of "CoRelation" 

# Environment
Using requriemnts.txt to install the environment.

# Dataset

Obtain licences to download MIMIC-III and MIMIC-IV datasets.

Once you obtain the MIMIC-III dataset, please follow [caml-mimic](https://github.com/jamesmullenbach/caml-mimic) to preprocess the dataset.

Next follow [MIMIC-IV-ICD](https://github.com/thomasnguyen92/MIMIC-IV-ICD-data-processing) to process the MIMIC-IV data.

Finally, follow [ICD-MSMN](https://github.com/GanjinZero/ICD-MSMN/tree/master) to obtain the final json format files for our model.

Using get_code_relation.py to obtain the ICD9 code relation graph.

Using get_code_relation10.py to obtain the ICD10 code relation graph.

# Word embedding
Please download [word2vec_sg0_100.model](https://github.com/aehrc/LAAT/blob/master/data/embeddings/word2vec_sg0_100.model) from LAAT.
You need to change the path of word embedding.

# Use our code
MIMIC-III Full (4 GPUs):
```
accelerate launch --multi_gpu main.py --version mimic3 --combiner lstm --rnn_dim 512 --num_layers 1 --decoder CoRelationV4 --attention_head 1 --attention_head_dim 256 --attention_dim 512 --learning_rate 5e-4 --train_epoch 30 --batch_size 4 --eval_batch_size 1 --gradient_accumulation_steps 1 --xavier --main_code_loss_weight 0.0 --rdrop_alpha 5.0 --est_cls 1 --term_count 8 --word_embedding_path embedding/word2vec_sg0_100.model --head_pooling mean --text_pooling max --alpha_weight 0.01 --use_graph --topk_num 300 --output_base_dir ./outputs/
```

MIMIC-III 50:
```
accelerate launch main.py --version mimic3-50 --combiner lstm --rnn_dim 512 --num_layers 1 --decoder CoRelationV4 --attention_head 1 --attention_head_dim 256 --attention_dim 512 --learning_rate 5e-4 --train_epoch 40 --batch_size 16 --eval_batch_size 1 --gradient_accumulation_steps 1 --xavier --main_code_loss_weight 0.0 --rdrop_alpha 12.5 --est_cls 1 --term_count 8 --word_embedding_path embedding/word2vec_sg0_100.model --head_pooling mean --text_pooling max --alpha_weight 0.001 --use_graph --output_base_dir ./outputs/
```
# Supported datasets
MIMIC-III-Full mimic3

MIMIC-III-50 mimic3-50

MIMIC-IV-ICD9-Full mimic4

MIMIC-IV-ICD9-50 mimic4-50

MIMIC-IV-ICD10-Full mimic4_10

MIMIC-IV-ICD10-50 mimic4_10-50
# Evaluate checkpoints
Upon acceptance
```
[mimic3 checkpoint]()

[mimic3-50 checkpoint]()


