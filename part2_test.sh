export BEST_P_NDB="results/densenet121_sgd_patches_ndb_36/fold_0/best_checkpoint.pth"
export BEST_RAHMAN="results/densenet121_sgd_rahman_36/fold_0/best_checkpoint.pth"

torchrun main.py --test --model_name=densenet121 --model_weights_path=${BEST_P_NDB} --dataset_name=patches_ndb --dataset_path="data/ndb_ufes/patches" --train_size=0.8 --k_folds=5 --batch_size=32
torchrun main.py --test --model_name=densenet121 --model_weights_path=${BEST_P_NDB} --dataset_name=rahman --dataset_path="data/rahman" --train_size=0.8 --k_folds=5 --batch_size=32
torchrun main.py --test --model_name=densenet121 --model_weights_path=${BEST_RAHMAN} --dataset_name=patches_ndb --dataset_path="data/ndb_ufes/patches" --train_size=0.8 --k_folds=5 --batch_size=32
torchrun main.py --test --model_name=densenet121 --model_weights_path=${BEST_RAHMAN} --dataset_name=rahman --dataset_path="data/rahman" --train_size=0.8 --k_folds=5 --batch_size=32