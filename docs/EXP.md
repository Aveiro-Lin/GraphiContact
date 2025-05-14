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
python src/tools/train_graphi_contact_behave_SIMU.py --n_infers N
python src/tools/train_graphi_contact_damon_SIMU.py --n_infers N
python src/tools/train_graphi_contact_rich_SIMU.py --n_infers N
```
The `--n_infers` parameter specifies the value of `N`, which, as discussed in the paper, can be set to 2, 3, or 4 for ablation studies. The case of `N=1` corresponds to the default baseline model. As mentioned in the paper, the best results are achieved when the number of SIMU Modeling inferences reaches 4. For example, with the DAMON dataset, the command should be 
```bash
cd GraphiContact/
python src/tools/train_graphi_contact_damon_SIMU.py --n_infers 4
```
The trained model weights are eventually saved in the `ckpt/` directory.
