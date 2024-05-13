process simGen0 {
    publishDir path: "${params.publishDir}/${params.experimentId}/daughter_states/variant=${sim_data.getBaseName()}/seed=${initial_seed}/generation=${generation}/agent_id=${agent_id}",  pattern: "*.json"

    errorStrategy { (task.attempt <= maxRetries) ? 'retry' : 'ignore' }

    tag "${sim_data.getBaseName()}"

    input:
    path config
    tuple path(sim_data), val(initial_seed), val(generation)
    val sim_seed
    val agent_id

    output:
    path config, emit: config
    tuple path(sim_data), val(initial_seed), val(next_generation), emit: nextGen
    path 'daughter_state_0.json', emit: d0
    path 'daughter_state_1.json', emit: d1
    // This information is necessary to group simulations for analysis scripts
    // In order: variant sim_data, experiment ID, variant name, seed, generation, agent_id, experiment ID
    tuple path(sim_data), val(params.experimentId), val("${sim_data.getBaseName()}"), val(initial_seed), val(generation), val(agent_id), emit: metadata

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
        --variant ${sim_data.getBaseName()}
    """

    // Used to test workflow
    stub:
    next_generation = generation + 1
    """
    echo "$config $sim_data $initial_seed $generation" > daughter_state_0.json
    echo "$sim_seed" > daughter_state_1.json
    """
}

process sim {
    publishDir path: "${params.publishDir}/${params.experimentId}/daughter_states/variant=${sim_data.getBaseName()}/seed=${initial_seed}/generation=${generation}/agent_id=${agent_id}",  pattern: "*.json"

    errorStrategy { (task.attempt <= maxRetries) ? 'retry' : 'ignore' }

    tag "${sim_data.getBaseName()}"

    input:
    path config
    tuple path(sim_data), val(initial_seed), val(generation)
    path initial_state
    val sim_seed
    val agent_id

    output:
    path config, emit: config
    tuple path(sim_data), val(initial_seed), val(next_generation), emit: next_gen
    path 'daughter_state_0.json', emit: d0
    path 'daughter_state_1.json', emit: d1
    tuple path(sim_data), val(params.experimentId), val("${sim_data.getBaseName()}"), val(initial_seed), val(generation), val(agent_id), emit: metadata

    script:
    next_generation = generation + 1
    """
    # Create empty daughter states so workflow can continue even if sim fails
    touch daughter_state_0.json
    touch daughter_state_1.json
    PYTHONPATH=${params.projectRoot} python ${params.projectRoot}/ecoli/experiments/ecoli_master_sim.py \\
        -c $config \\
        --sim_data_path $sim_data \\
        --initial_state_file $initial_state \\
        --daughter_outdir \$(pwd) \\
        --variant ${sim_data.getBaseName()}
    """

    stub:
    next_generation = generation + 1
    """
    echo "$config $sim_data $initial_seed $generation" > daughter_state_0.json
    echo "$initial_state $sim_seed" > daughter_state_1.json
    """
}
