#!/bin/bash
#SBATCH --job-name=DifferLand               # Job name
#SBATCH --output=log_%x_%A_%a.out           # Standard output log
#SBATCH --error=log_%x_%A_%a.err            # Standard error log
#SBATCH --partition=node                    # Partition/queue name
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#BATCH --mem=20G

# Record start time
date

# Load Python environment
source /share/home/fyliu0903/anaconda3/etc/profile.d/conda.sh
conda activate DifferLand

# Execute the Python script
python -u calibration.py "$@"

# Record end time
date
