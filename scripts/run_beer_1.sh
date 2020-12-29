dataset="beer"
aspect=1
balance="True"
output_dir="./beer_results/aspect"$aspect

data_dir="./data/beer_review/beer"$aspect
annotation_path="./data/beer_review/annotations.json"
embedding_dir="./embeddings"
embedding_name="glove.6B.100d.txt"

batch_size=200
visual_interval=10000

#selective parameters
sparsity_percentage=0.06
sparsity_lambda=2.
continuity_lambda=10.

cls_lambda=0.1
om_lambda=0.9
fm_lambda=0.01
seed=12252020
pretrain_epchos=4
num_epchos=22
gpu=7

/data1/share/harden/anaconda3/bin/python run.py --dataset $dataset \
--data_dir $data_dir --balance $balance --aspect $aspect --output_dir $output_dir --embedding_dir $embedding_dir --embedding_name $embedding_name --annotation_path $annotation_path  --batch_size $batch_size  \
--num_epchos $num_epchos --pretrain_epchos $pretrain_epchos \
--seed $seed \
--cls_lambda $cls_lambda --om_lambda $om_lambda --fm_lambda $fm_lambda \
--sparsity_percentage $sparsity_percentage --sparsity_lambda $sparsity_lambda  --continuity_lambda $continuity_lambda \
--visual_interval $visual_interval --gpu $gpu 