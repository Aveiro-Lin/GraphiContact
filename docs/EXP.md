# Training & Testing

## Original Model

We use the following scripts to train and test on the three datasets you have downloaded. 

```bash
bash GraphiContact/run_behave.sh
bash GraphiContact/run_damon.sh
bash GraphiContact/run_rich.sh
```

## MIMO Model
Adjust the original model architecture[GraphiContact/src/modeling/bert/e2e_body_network.py] to the MIMO model[GraphiContact/src/modeling/bert/e2e_body_network(mimo).py]

```bash
python GraphiContact/src/tools/train_deco_contact_behave（mimo）.py
python GraphiContact/src/tools/train_deco_contact_damon（mimo）.py
python GraphiContact/src/tools/train_deco_contact_rich（mimo）.py
'''

