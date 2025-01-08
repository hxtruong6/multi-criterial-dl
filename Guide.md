# Guide to setup

## Dataset

URL: <https://drive.google.com/drive/folders/1bKPVFBbk8zjwrUelORKXh5MECHGrZpJp?usp=drive_link>

## Environment

Step to create a conda environment for self-explain module:

```bash
cd selfexplain
conda create --name research_selfexplain python=3.11
conda activate research_selfexplain
# Install Pytorch
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install the requirements
pip install -r requirements.txt


```

## Kagayaki

```bash
# Access the Kagayaki Server:
ssh s2320437@kagayaki
qsub −q DEFAULT −I

# In case of using GPU to compute your code:
ssh s2320437@kagayaki
qsub −q GPU−1 −I

# Show all module which can use
module avail
# Tell Kagayaki gives you CUDA Tool Kit
module load cuda/12.1

# Show all current modules loaded
module list

# Check Nvida stats (optional)
nvidia-smi
```
Example of using GPU:

```bash
# (base) s2320437@kagayaki ~/WORK/multi-criterial-dl/selfexplain ±main » 
qsub -q GPU-1 -I                                                                                                                                                                                                       130 ↵
#qsub: waiting for job 12335257.spcc-adm1 to start
#qsub: job 12335257.spcc-adm1 ready

# (base) s2320437@spcc-a40g06 ~/WORK » 
```
