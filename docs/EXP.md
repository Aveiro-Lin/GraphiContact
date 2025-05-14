# Training

## Original Model

We use the following scripts to train and test on the three datasets you have downloaded. 

```bash
cd GraphiContact/
bash run_behave.sh
bash run_damon.sh
bash run_rich.sh
```
The trained model weights are ultimately saved under the `ckpt/` directory.

## SIMU Model

```bash
cd GraphiContact/
python src/tools/train_graphi_contact_behave_SIMU.py --n_infers [n]
python src/tools/train_graphi_contact_damon_SIMU.py --n_infers [n]
python src/tools/train_graphi_contact_rich_SIMU.py --n_infers [n]
```
The `--n_infers` parameter specifies the value of `[n]`, which, as discussed in the paper, can be set to 2, 3, or 4 for ablation studies. The case of `n=1` corresponds to the default baseline model. The trained model weights are eventually saved in the `ckpt/` directory.
