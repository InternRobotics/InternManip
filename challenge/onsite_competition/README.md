# 🧭 IROS On-site Challenge

Welcome to the **IROS 2025 Challenge — Track: Vision-Language Manipulation in Open Tabletop Environments**!  
In this phase, participants’ models will be deployed on **a real robot** to evaluate performance in real-world conditions.

---
Based on the [guidelines](../guidelines.md) from the previous online competition, we enhanced the dataset by incorporating real-world training data and provided example scripts for both training and testing.

## 📦 Real-World Training Data Supplement
The data remains in the same Hugging Face repository used for the online competition. Please download the data following the instructions in [Prepare dataset](../guidelines.md#prepare-train--validation-dataset).  
> Note that a new folder named `train_real` has been added under `./data/dataset/IROS-2025-Challenge-Manip/`.

## 🛠️ Train
Based on the previous training tutorial in [Train](../guidelines.md#train), we have added an example [configuration file](../../run_configs/train/gr00t_n1_5_arx_iros.yaml) for training the `gr00t n1.5` on real-world datasets, provided as a reference.

You can use the following command to start the training.
```bash
torchrun --nnodes 1 --nproc_per_node 8 scripts/train/train.py --config run_configs/train/gr00t_n1_5_arx_iros.yaml
```

## ✅ Evaluation

You can verify the agent’s observation and action spaces by running a dummy test to ensure the I/O data format between the environment and the agent is aligned.  
Run the following command to perform this test.

```bash
python -m scripts.eval.start_agent_server

# Wait for the server to start.

python -m challenge.scripts.start_dummy_evaluator --config challenge/run_configs/eval/gr00t_n1_5_on_real_dummy.py --server
```


## 📋 Rules
Please check out the [onsite competition rules](./onsite_competition_rules_en-US.md).


## 🚀 Code Submission
### Key Requirements
All competition images must be uploaded to the public image repository by **October 18**. Each image must contain a pre-configured and operational agent server.
  
### Image Usage Policy
- **Debugging Phase (Oct 18)**: Participants may submit multiple images for testing purposes.
  
- **Main Competition (Oct 19)**: Only **one** submitted image be used during the official competition.
  
### Recommended Practice
To ensure smooth competition execution, we strongly recommend participants store their submitted images on portable USB drives as offline backups. Bringing these physical copies to the venue will prevent potential delays caused by network issues during image retrieval.
