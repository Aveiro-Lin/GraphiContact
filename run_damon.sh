CURRENT_DIR=$(cd $(dirname $0); pwd)
echo "CURRENT_DIR: "$CURRENT_DIR
CUDA_VISIBLE_DEVICES=1 python $CURRENT_DIR/src/tools/train_graphi_contact_damon.py

# The above instructions do not include using the SIMU architecture. If you want to use the SIMU Architecture, please use the following command:

# python $CURRENT_DIR/src/tools/train_graphi_contact_damon_SIMU.py --n_infers n

# Here, "n" refers to the number of parallel streams. For example, if the optimal number of streams mentioned in the paper is N=4, the training command would look like this:

# python $CURRENT_DIR/src/tools/train_graphi_contact_damon_SIMU.py --n_infers 4
