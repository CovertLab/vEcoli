#! /usr/bin/env bash

# Runs the parca and sims twice to compare output to ensure both
# are reproducible from run to run. Will exit with an error code
# if either the parca or sim runs produce different output from
# each other.

set -eu

script_dir=$(dirname $(dirname $(realpath $0)))
out_dir=$(dirname $script_dir)/out
data_dir=$(dirname $script_dir)/data
dir1="check-reproducibility-1"
dir2="check-reproducibility-2"
git_hash=$(git rev-parse HEAD)

source $script_dir/jenkins/setup-environment.sh

# Run parca twice and check that output is consistent from run to run
python $script_dir/parca.py -c 4 -o "$out_dir/$dir1-$git_hash"
python $script_dir/parca.py -c 4 -o "$out_dir/$dir2-$git_hash"
python $script_dir/debug/compare_pickles.py "$out_dir/$dir1-$git_hash/kb" "$out_dir/$dir2-$git_hash/kb"

# Run entire simulation for each parca output
python $(dirname $script_dir)/ecoli/experiments/ecoli_master_sim.py \
    --config $script_dir/jenkins/configs/reproducibility.json \
    --experiment_id "$dir1-$git_hash" --daughter_outdir "$out_dir/$dir1-$git_hash" \
    --sim_data_path "$out_dir/$dir1-$git_hash/kb/simData.cPickle" --fail_at_total_time
python $(dirname $script_dir)/ecoli/experiments/ecoli_master_sim.py \
    --config $script_dir/jenkins/configs/reproducibility.json \
    --experiment_id "$dir2-$git_hash" --daughter_outdir "$out_dir/$dir2-$git_hash" \
    --sim_data_path "$out_dir/$dir2-$git_hash/kb/simData.cPickle" --fail_at_total_time

# Run short daughters to check that everything including division is consistent
ln -s "$out_dir/$dir1-$git_hash/daughter_state_0.json" $data_dir/daughter_state_0.json
python $(dirname $script_dir)/ecoli/experiments/ecoli_master_sim.py \
    --config $script_dir/jenkins/configs/reproducibility.json \
    --experiment_id "$dir1-$git_hash" --agent_id "00" --total_time 10 \
    --initial_state_file daughter_state_0
rm $data_dir/daughter_state_0.json
ln -s "$out_dir/$dir2-$git_hash/daughter_state_0.json" $data_dir/daughter_state_0.json
python $(dirname $script_dir)/ecoli/experiments/ecoli_master_sim.py \
    --config $script_dir/jenkins/configs/reproducibility.json \
    --experiment_id "$dir2-$git_hash" --agent_id "00" --total_time 10 \
    --initial_state_file daughter_state_0
python $script_dir/debug/diff_simouts.py -o "$out_dir" "$dir1-$git_hash" "$dir2-$git_hash"
