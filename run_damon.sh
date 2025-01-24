CUDA_VISIBLE_DEVICES=1 python src/tools/train_deco_contact_damon.py

# The above instructions do not include using the SMM architecture. If you want to use the SMM Architecture, please use the following command:

# python src/tools/train_deco_contact_damon_smm.py --n_infers n

# Here, "n" refers to the number of parallel streams. For example, if the optimal number of streams mentioned in the paper is N=4, the training command would look like this:

# python src/tools/train_deco_contact_damon_smm.py --n_infers 4
