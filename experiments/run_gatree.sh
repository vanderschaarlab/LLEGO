DATASET_LIST=(credit-g diabetes compas heart-statlog liver breast vehicle)
MAX_DEPTH_LIST=(3 4)
SEED_LIST=(0 1 2 3 4)
exp_name='classification'


for seed in "${SEED_LIST[@]}"
do
    for max_depth in "${MAX_DEPTH_LIST[@]}"
    do 
        for dataset in "${DATASET_LIST[@]}"
        do
            echo "Running experiment for dataset: $dataset, max_depth: $max_depth, model: $model and exp_name: $exp_name"
            python exp_gatree.py log_wandb=True dataset=$dataset max_depth=$max_depth exp_name="$exp_name" seed=$seed
        done
    done
done


DATASET_LIST=(cholesterol wine wage abalone cars)
exp_name='regression'


for seed in "${SEED_LIST[@]}"
do
    for max_depth in "${MAX_DEPTH_LIST[@]}"
    do 
        for dataset in "${DATASET_LIST[@]}"
        do
            echo "Running experiment for dataset: $dataset, max_depth: $max_depth, model: $model and exp_name: $exp_name"
            python exp_gatree.py log_wandb=True dataset=$dataset max_depth=$max_depth exp_name="$exp_name" seed=$seed
        done
    done
done






