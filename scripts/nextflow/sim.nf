process simGen0 {
    publishDir path: "${params.publishDir}/${params.experimentId}/daughter_states/variant=${sim_data.getBaseName()}/seed=${lineage_seed}/generation=${generation}/agent_id=${agent_id}",  pattern: "*.json"

    errorStrategy { (task.attempt <= maxRetries) ? 'retry' : 'ignore' }

    tag "${sim_data.getBaseName()}"

    input:
    path config
    tuple path(sim_data), val(lineage_seed), val(generation)
    val sim_seed
    val agent_id

    output:
    path config, emit: config
    tuple path(sim_data), val(lineage_seed), val(next_generation), emit: nextGen
    path 'daughter_state_0.json', emit: d0
    path 'daughter_state_1.json', emit: d1
    // This information is necessary to group simulations for analysis scripts
    // In order: variant sim_data, experiment ID, variant name, seed, generation, agent_id, experiment ID
    tuple path(sim_data), val(params.experimentId), val("${sim_data.getBaseName()}"), val(lineage_seed), val(generation), val(agent_id), emit: metadata

    script:
    next_generation = generation + 1
    """
    # Create empty daughter states so workflow can continue even if sim fails
    touch daughter_state_0.json
    touch daughter_state_1.json
    PYTHONPATH=${params.projectRoot} python ${params.projectRoot}/ecoli/experiments/ecoli_master_sim.py \\
        -c $config \\
        --sim_data_path $sim_data \\
        --daughter_outdir \$(pwd) \\
        --variant ${sim_data.getBaseName()} \\
        --seed ${sim_seed} \\
        --lineage_seed ${lineage_seed}
    """

    // Used to test workflow
    stub:
    next_generation = generation + 1
    """
    echo "$config $sim_data $lineage_seed $generation" > daughter_state_0.json
    echo "$sim_seed" > daughter_state_1.json
    """
}

process sim {
    publishDir path: "${params.publishDir}/${params.experimentId}/daughter_states/variant=${sim_data.getBaseName()}/seed=${lineage_seed}/generation=${generation}/agent_id=${agent_id}",  pattern: "*.json"

    errorStrategy { (task.attempt <= maxRetries) ? 'retry' : 'ignore' }

    tag "${sim_data.getBaseName()}"

    input:
    path config
    tuple path(sim_data), val(lineage_seed), val(generation)
    path initial_state, stageAs: 'data/*'
    val sim_seed
    val agent_id

    output:
    path config, emit: config
    tuple path(sim_data), val(lineage_seed), val(next_generation), emit: nextGen
    path 'daughter_state_0.json', emit: d0
    path 'daughter_state_1.json', emit: d1
    tuple path(sim_data), val(params.experimentId), val("${sim_data.getBaseName()}"), val(lineage_seed), val(generation), val(agent_id), emit: metadata

    script:
    next_generation = generation + 1
    """
    # Create empty daughter states so workflow can continue even if sim fails
    touch daughter_state_0.json
    touch daughter_state_1.json
    PYTHONPATH=${params.projectRoot} python ${params.projectRoot}/ecoli/experiments/ecoli_master_sim.py \\
        -c $config \\
        --sim_data_path $sim_data \\
        --initial_state_file ${initial_state.getBaseName()} \\
        --daughter_outdir \$(pwd) \\
        --variant ${sim_data.getBaseName()} \\
        --seed ${sim_seed} \\
        --lineage_seed ${lineage_seed} \\
        --agent_id \'${agent_id}\'
    """

    stub:
    next_generation = generation + 1
    """
    echo "$config $sim_data $lineage_seed $generation" > daughter_state_0.json
    echo "$initial_state $sim_seed" > daughter_state_1.json
    """
}
