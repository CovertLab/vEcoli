process simGen0 {
    publishDir path: "${params.publishDir}/${params.experimentId}/daughter_states/variant=${sim_data.getBaseName()}/seed=${lineage_seed}/generation=${generation}/agent_id=${agent_id}",  pattern: "daughter_state_*.json", mode: "copy"

    tag "variant=${sim_data.getBaseName()}/seed=${lineage_seed}/generation=${generation}/agent_id=${agent_id}"

    input:
    path config
    tuple path(sim_data), val(lineage_seed), val(generation)
    val agent_id

    output:
    tuple path(config), path(sim_data), val(lineage_seed), val(next_generation), val(seed_d0), path('daughter_state_0.json'), val(agent_id_d0), env(division_time), emit: nextGen0
    tuple path(config), path(sim_data), val(lineage_seed), val(next_generation), val(seed_d1), path('daughter_state_1.json'), val(agent_id_d1), env(division_time), emit: nextGen1
    // This information is necessary to group simulations for analysis scripts
    // In order: variant sim_data, experiment ID, variant name, seed, generation, agent_id, experiment ID
    tuple path(sim_data), val(params.experimentId), val("${sim_data.getBaseName()}"), val(lineage_seed), val(generation), val(agent_id), emit: metadata

    script:
    next_generation = generation + 1
    agent_id_d0 = agent_id + '0'
    agent_id_d1 = agent_id + '1'
    seed_d0 = lineage_seed + 1
    seed_d1 = lineage_seed + 2
    """
    # Create empty daughter states so workflow can continue even if sim fails
    touch daughter_state_0.json
    touch daughter_state_1.json
    touch division_time.sh
    # Use 1 Polars thread to avoid oversubscription on HPC/cloud
    POLARS_MAX_THREADS=1 python ${params.projectRoot}/ecoli/experiments/ecoli_master_sim.py \\
        --config $config \\
        --sim_data_path $sim_data \\
        --daughter_outdir "\$(pwd)" \\
        --variant ${sim_data.getBaseName()} \\
        --seed ${lineage_seed} \\
        --lineage_seed ${lineage_seed} \\
        --agent_id \'${agent_id}\'
    source division_time.sh
    """

    // Used to test workflow
    stub:
    next_generation = generation + 1
    agent_id_d0 = agent_id + '0'
    agent_id_d1 = agent_id + '1'
    seed_d0 = sim_seed + 1
    seed_d1 = sim_seed + 2
    """
    echo "$config $sim_data $lineage_seed $generation" > daughter_state_0.json
    echo "$sim_seed" > daughter_state_1.json
    export division_time=1000
    """
}

process sim {
    publishDir path: "${params.publishDir}/${params.experimentId}/daughter_states/variant=${sim_data.getBaseName()}/seed=${lineage_seed}/generation=${generation}/agent_id=${agent_id}",  pattern: "daughter_state_*.json", mode: "copy"

    tag "variant=${sim_data.getBaseName()}/seed=${lineage_seed}/generation=${generation}/agent_id=${agent_id}"

    input:
    tuple path(config), path(sim_data), val(lineage_seed), val(generation), val(sim_seed), path(initial_state, stageAs: 'data/*'), val(agent_id), val(prev_division_time)

    output:
    tuple path(config), path(sim_data), val(lineage_seed), val(next_generation), val(seed_d0), path('daughter_state_0.json'), val(agent_id_d0), env(division_time), emit: nextGen0
    tuple path(config), path(sim_data), val(lineage_seed), val(next_generation), val(seed_d1), path('daughter_state_1.json'), val(agent_id_d1), env(division_time), emit: nextGen1
    tuple path(sim_data), val(params.experimentId), val("${sim_data.getBaseName()}"), val(lineage_seed), val(generation), val(agent_id), emit: metadata

    script:
    next_generation = generation + 1
    agent_id_d0 = agent_id + '0'
    agent_id_d1 = agent_id + '1'
    seed_d0 = sim_seed + 1
    seed_d1 = sim_seed + 2
    """
    # Create empty daughter states so workflow can continue even if sim fails
    touch daughter_state_0.json
    touch daughter_state_1.json
    touch division_time.sh
    # Use 1 Polars thread to avoid oversubscription on HPC/cloud
    POLARS_MAX_THREADS=1 python ${params.projectRoot}/ecoli/experiments/ecoli_master_sim.py \\
        --config $config \\
        --sim_data_path $sim_data \\
        --initial_state_file ${initial_state.getBaseName()} \\
        --daughter_outdir "\$(pwd)" \\
        --variant ${sim_data.getBaseName()} \\
        --seed ${sim_seed} \\
        --lineage_seed ${lineage_seed} \\
        --agent_id \'${agent_id}\' \\
        --initial_global_time ${prev_division_time}
    source division_time.sh
    """

    stub:
    next_generation = generation + 1
    agent_id_d0 = agent_id + '0'
    agent_id_d1 = agent_id + '1'
    seed_d0 = sim_seed + 1
    seed_d1 = sim_seed + 2
    """
    echo "$config $sim_data $lineage_seed $generation" > daughter_state_0.json
    echo "$initial_state $sim_seed" > daughter_state_1.json
    export division_time=1000
    """
}
