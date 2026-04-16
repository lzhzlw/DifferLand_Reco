#!/bin/bash

# Define station list
stations=("US-Var" "FR-Pue" "NL-Loo" "US-Ha1")

# Define parameter values for m and w
m_values=(1 2)
w_values=(0 1)
# Create a directory for log files
mkdir -p logs

# Iterate over stations and parameter combinations
for station in "${stations[@]}"; do
    for m in "${m_values[@]}"; do
        for w in "${w_values[@]}"; do
            
            # Define the job name
            job_name="${station}_m${m}_w${w}"

            # Submit the job using sbatch
            sbatch --job-name=${job_name} \
                   --output=logs/${job_name}_%j.log \
                   --error=logs/${job_name}_%j.err \
                   --partition=node \
                   --nodes=1 \
                   --nodelist=node04 \
                   --wrap="date; \
                           module load python; \
                           python -u /data1/fyliu/e_soil_carbon_model/code/DifferLand_v3_1_site_soil_pool/experiments/calibration.py -n ${station} -d ${station}.nc -c 2 -i 2 -m ${m} -w ${w}; \
                           date"
        done
    done
done