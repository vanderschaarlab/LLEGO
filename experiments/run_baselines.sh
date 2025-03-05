DATASET_LIST=(credit-g diabetes compas heart-statlog liver breast vehicle)
MAX_DEPTH_LIST=(3 4)
MODEL_LIST=(cart gosdt dl85 c45)
SEED_LIST=(0 1 2 3 4)
exp_name='classification'
metric_name="balanced_accuracy"

for seed in "${SEED_LIST[@]}"
do
    for max_depth in "${MAX_DEPTH_LIST[@]}"
    do 
        for dataset in "${DATASET_LIST[@]}"
        do
            for model in "${MODEL_LIST[@]}"
            do
                if [ "$model" == "cart" ] || [ "$model" == "c45" ]; then
                    n_trials_hpt=20
                else # Optimal methods (GOSDT and DL85) take much longer to run
                    n_trials_hpt=10
                fi

                echo "Running experiment for dataset: $dataset, max_depth: $max_depth, model: $model and exp_name: $exp_name"
                python exp_baselines.py dataset="$dataset" max_depth=$max_depth exp_name="$exp_name" n_trials_hpt=$n_trials_hpt baseline=$model seed=$seed metric_name=$metric_name
            done
    
        done
    done
done



DATASET_LIST=(cholesterol wine wage abalone cars)
MODEL_LIST=(cart_reg) # Only CART supports regression
exp_name='regression'
metric_name='mse'

for seed in "${SEED_LIST[@]}"
do
    for max_depth in "${MAX_DEPTH_LIST[@]}"
    do 
        for dataset in "${DATASET_LIST[@]}"
        do
            for model in "${MODEL_LIST[@]}"
            do
                if [ "$model" == "cart" ] || [ "$model" == "c45" ]; then
                    n_trials_hpt=20
                else # Optimal methods (GOSDT and DL85) take much longer to run
                    n_trials_hpt=10
                fi

                echo "Running experiment for dataset: $dataset, max_depth: $max_depth, model: $model and exp_name: $exp_name"
                python exp_baselines.py dataset="$dataset" max_depth=$max_depth exp_name="$exp_name" n_trials_hpt=$n_trials_hpt baseline=$model seed=$seed metric_name=$metric_name
            done
    
        done
    done
done


