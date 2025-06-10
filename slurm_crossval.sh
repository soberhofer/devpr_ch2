#!/bin/bash

#SBATCH --job-name=training      # Job name
#SBATCH --output=logs/esc50_training_%j.out # Standard output and error log (%j expands to jobID) - recommend a logs subdir
#SBATCH --error=logs/esc50_training_%j.err  # Standard error log - recommend a logs subdir
#SBATCH --nodes=1                      # Run on a single node
#SBATCH --ntasks-per-node=1            # Run one task
#SBATCH --cpus-per-task=4              # Number of CPUs per task (adjust if needed)
#SBATCH --mem=0                      # Memory per node (adjust if needed)
#SBATCH --export=WANDB_API_KEY

# --- Create Log Directory ---
# It's good practice to have Slurm output/error logs in a dedicated directory.
# This command will create 'logs' in the directory where you submit the job.
mkdir -p logs
export OMP_NUM_THREADS=1
# --- Environment Setup ---
echo "========= ENVIRONMENT SETUP START ========="
date
echo "Job ID: $SLURM_JOB_ID"
echo "Running on host: $(hostname)"
echo "Working directory: $(pwd)" # This will be the directory where you sbatch from

# Load Conda
# Replace with the correct path to your Conda initialization script if different.
# Common paths: ~/anaconda3/etc/profile.d/conda.sh, /opt/conda/etc/profile.d/conda.sh
CONDA_INIT_SCRIPT="/opt/miniconda3/etc/profile.d/conda.sh" # <<< YOU MIGHT NEED TO ADJUST THIS
if [ -f "$CONDA_INIT_SCRIPT" ]; then
    source "$CONDA_INIT_SCRIPT"
    echo "Sourced Conda init script: $CONDA_INIT_SCRIPT"
else
    echo "ERROR: Conda initialization script not found at $CONDA_INIT_SCRIPT"
    exit 1
fi

# Define Conda environment name and Python version 
CONDA_ENV_NAME="ai24m007" 
PYTHON_VERSION="3.11" # As you specified 
# Check if the environment already exists 
if conda info --envs | grep -q "^${CONDA_ENV_NAME}\s"; then 
    echo "Conda environment '${CONDA_ENV_NAME}' already exists." 
else
    echo "Conda environment '${CONDA_ENV_NAME}' does not exist. Creating it with Python ${PYTHON_VERSION}..." 
    conda create -n "${CONDA_ENV_NAME}" python="${PYTHON_VERSION}" -y 
    if [ $? -ne 0 ]; then 
        echo "ERROR: Failed to create Conda environment '${CONDA_ENV_NAME}'." 
        exit 1
    fi
    echo "Conda environment '${CONDA_ENV_NAME}' created successfully." 
fi


# Activate your Conda environment
# Replace 'your_pytorch_env' with the name of your Conda environment.
# If you don't have one, create it first (e.g., manually on the login node).
conda activate "${CONDA_ENV_NAME}"
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to activate Conda environment: ${CONDA_ENV_NAME}"
    echo "Please ensure the environment exists and is properly configured."
    exit 1
fi

# Verify activation
CURRENT_ENV=$(conda info --envs | grep '*' | awk '{print $1}')
if [ "${CURRENT_ENV}" != "${CONDA_ENV_NAME}" ]; then
    echo "ERROR: Conda environment '${CONDA_ENV_NAME}' did not activate correctly. Current env is '${CURRENT_ENV}'."
    exit 1
fi
echo "Successfully activated Conda environment: ${CONDA_ENV_NAME}"
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available via PyTorch: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "CUDA device count visible to PyTorch: $(python -c 'import torch; print(torch.cuda.device_count())')"
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    echo "Slurm assigned CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
else
    echo "CUDA_VISIBLE_DEVICES is not set by Slurm (or not visible here)."
fi
#echo "WANDB_API_KEY is set: $(if [ -n "$WANDB_API_KEY" ]; then echo "Yes"; else echo "No - ensure it is in .bashrc and inherited";exit 1; fi)"
echo "========= ENVIRONMENT SETUP END ========="
echo

# --- Code Setup on Compute Node ---
# Using $SLURM_TMPDIR for temporary files is good practice if available and configured on your cluster.
# Otherwise, create a job-specific directory in your scratch space or a shared project space.
# This example uses a directory within the submission directory for simplicity,
# but $SLURM_TMPDIR or a scratch space is generally better for performance and cleanup.
JOB_WORK_DIR="${SLURM_SUBMIT_DIR}/slurm_job_data_${SLURM_JOB_ID}"
mkdir -p "${JOB_WORK_DIR}"
cd "${JOB_WORK_DIR}"
echo "Changed to job working directory: $(pwd)"

echo "Cloning repository devpr_ch2..."
git clone https://github.com/soberhofer/devpr_ch2 Repo
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to clone repository."
    exit 1
fi
cd Repo
#git checkout inmemdata
#git checkout -b "ResNet18-fold5-2123-20250520_191227" bbc2ab307cc72559da55bf420760beb7d41691f5
echo "Changed directory to $(pwd)" # Should be JOB_WORK_DIR/Repo

echo "Installing Python requirements..."
#pip install torch==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install requirements."
    exit 1
fi
echo "Python requirements installed."
echo
#python preprocess_augment_data.py paths.original_data_root=../data/esc50 settings.num_augmentations_per_file=8

# --- Run Training ---
echo "========= TRAINING START ========="
date

# Capture the first argument passed to this sbatch script
# This argument will contain the Hydra overrides for this specific run
HYDRA_OVERRIDES="$1"

if [ -z "$HYDRA_OVERRIDES" ]; then
    echo "WARNING: No Hydra overrides provided as an argument to the sbatch script."
    #echo "Usage: sbatch run_training_slurm.sbatch \"your.hydra.param=value another.param=value\""
    #exit 1
fi
echo "Starting training script with Hydra overrides: ${HYDRA_OVERRIDES}"

# The WANDB_API_KEY should be inherited from your .bashrc.
# Hydra parameters are passed directly.
# Note: data.test_folds='[1]' is quoted to ensure Hydra receives it as a list string.
python -u train_crossval.py \
    ${HYDRA_OVERRIDES} \
    comment=$SLURM_JOB_ID
    # Add any other specific Hydra overrides you used in Colab, for example:
    # model.params.dropout_prob=0.5 \
    # training.scheduler.max_lr=1e-2 \
    # training.device_id=0 # This is default, but can be explicit
    # Ensure that parameters in HYDRA_OVERRIDES take precedence or are distinct
    # from any fixed parameters listed here. Hydra handles overrides well.

if [ $? -ne 0 ]; then
    echo "ERROR: Training script exited with an error."
    # Consider adding logic here to copy partial results if needed
    exit 1
fi
date
echo "========= TRAINING END ========="
echo
python test_crossval.py ${HYDRA_OVERRIDES}
# --- Post-Training (Optional) ---
# You might want to copy important results from JOB_WORK_DIR/Repo/outputs (Hydra's default)
# back to your SLURM_SUBMIT_DIR or another persistent location.
# Example:
# echo "Copying Hydra outputs..."
# mkdir -p "${SLURM_SUBMIT_DIR}/results_${SLURM_JOB_ID}"
# cp -r outputs/* "${SLURM_SUBMIT_DIR}/results_${SLURM_JOB_ID}/"
# echo "Outputs copied to ${SLURM_SUBMIT_DIR}/results_${SLURM_JOB_ID}"

# --- Cleanup (Optional but Recommended) ---
# If using $SLURM_TMPDIR, it's often cleaned automatically.
# If you created JOB_WORK_DIR in a shared space, clean it up.
# echo "Cleaning up job working directory: ${JOB_WORK_DIR}"
# rm -rf "${JOB_WORK_DIR}"

echo "Slurm job finished."
date
