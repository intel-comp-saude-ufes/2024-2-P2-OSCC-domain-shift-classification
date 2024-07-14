export BEST_P_NDB="results/train/densenet121_sgd_patches_ndb_5/fold_2/best_checkpoint.pth"
export BEST_RAHMAN="results/train/densenet121_sgd_rahman_6/fold_0/best_checkpoint.pth"

torchrun main.py --test --model_name=densenet121 --model_weights_path=${BEST_P_NDB} --dataset_name=patches_ndb --dataset_path="data/ndb_ufes/patches" --train_size=0.8 --k_folds=5 --batch_size=32 --run_name="Patches Part2 - P_NDB with P_NDB weights"
torchrun main.py --test --model_name=densenet121 --model_weights_path=${BEST_P_NDB} --dataset_name=rahman --dataset_path="data/rahman" --train_size=0.8 --k_folds=5 --batch_size=32 --run_name="Patches Part2 - RAHMAN with P_NDB weights"
torchrun main.py --test --model_name=densenet121 --model_weights_path=${BEST_RAHMAN} --dataset_name=patches_ndb --dataset_path="data/ndb_ufes/patches" --train_size=0.8 --k_folds=5 --batch_size=32 --run_name="Patches Part2 - P_NDB with RAHMAN weights"
torchrun main.py --test --model_name=densenet121 --model_weights_path=${BEST_RAHMAN} --dataset_name=rahman --dataset_path="data/rahman" --train_size=0.8 --k_folds=5 --batch_size=32 --run_name="Patches Part2 - RAHMAN with RAHMAN weights"