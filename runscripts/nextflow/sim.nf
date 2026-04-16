process simGen0 {
    // Daughter states are written directly to cloud storage via fsspec

    label "sim"

    tag { "variant=${variant}/seed=${lineage_seed}/generation=${generation}/agent_id=${agent_id}" }

    input:
    // Accept URIs and hashes (no path staging - files accessed via cloud URIs)
    tuple val(config_uri), val(config_hash), val(sim_data_uri), val(sim_data_hash), val(variant), val(lineage_seed), val(generation)
    val agent_id

    output:
    // Pass daughter state URIs as val() strings (no file staging)
    tuple val(config_uri), val(config_hash), val(sim_data_uri), val(sim_data_hash), val(variant), val(lineage_seed), val(next_generation), val(seed_d0), env('daughter_state_0_uri'), val(agent_id_d0), env('division_time'), emit: nextGen0
    tuple val(config_uri), val(config_hash), val(sim_data_uri), val(sim_data_hash), val(variant), val(lineage_seed), val(next_generation), val(seed_d1), env('daughter_state_1_uri'), val(agent_id_d1), env('division_time'), emit: nextGen1
    // This information is necessary to group simulations for analysis scripts
    // In order: variant sim_data URI, sim_data_hash (for cache invalidation), experiment ID, variant name, seed, generation, agent_id
    tuple val(sim_data_uri), val(sim_data_hash), val(params.experimentId), val(variant), val(lineage_seed), val(generation), val(agent_id), emit: metadata

    script:
    next_generation = generation + 1
    agent_id_d0 = agent_id + '0'
    agent_id_d1 = agent_id + '1'
    seed_d0 = lineage_seed + 1
    seed_d1 = lineage_seed + 2
    def daughter_outdir = "${params.publishDir}/${params.experimentId}/daughter_states/variant=${variant}/seed=${lineage_seed}/generation=${generation}/agent_id=${agent_id}"
    """
    # Belt-and-suspenders: source /vEcoli/.env so ECOLI_SOURCES resolves
    # correctly even when the container's ENTRYPOINT is bypassed.
    [[ -f /vEcoli/.env ]] && set -a && source /vEcoli/.env && set +a || true
    # Use 1 Polars thread to avoid oversubscription on HPC/cloud
    # Access config and sim_data directly via cloud URIs (fsspec handles both S3 and GCS)
    # Daughter states are written directly to cloud storage via fsspec
    PYTHONUNBUFFERED=1 POLARS_MAX_THREADS=1 python ${params.projectRoot}/ecoli/experiments/ecoli_master_sim.py \\
        --config "${config_uri}" \\
        --sim_data_path "${sim_data_uri}" \\
        --daughter_outdir "${daughter_outdir}" \\
        --variant ${variant} \\
        --seed ${lineage_seed} \\
        --lineage_seed ${lineage_seed} \\
        --agent_id \'${agent_id}\'
    # Set division_time environment variable from file written by ecoli_master_sim.py
    source division_time.sh
    # Read daughter state URIs from files written by ecoli_master_sim.py
    export daughter_state_0_uri=\$(cat daughter_state_0_uri.txt)
    export daughter_state_1_uri=\$(cat daughter_state_1_uri.txt)
    """

    // Used to test workflow
    stub:
    next_generation = generation + 1
    agent_id_d0 = agent_id + '0'
    agent_id_d1 = agent_id + '1'
    seed_d0 = lineage_seed + 1
    seed_d1 = lineage_seed + 2
    def daughter_outdir = "${params.publishDir}/${params.experimentId}/daughter_states/variant=${variant}/seed=${lineage_seed}/generation=${generation}/agent_id=${agent_id}"
    """
    mkdir -p ${daughter_outdir}
    echo "$config_uri $sim_data_uri $lineage_seed $generation" > ${daughter_outdir}/daughter_state_0.json
    echo "$lineage_seed" > ${daughter_outdir}/daughter_state_1.json
    export daughter_state_0_uri=${daughter_outdir}/daughter_state_0.json
    export daughter_state_1_uri=${daughter_outdir}/daughter_state_1.json
    export division_time=1000
    """
}

process sim {
    // Daughter states are written directly to cloud storage via fsspec

    label "sim"

    tag { "variant=${variant}/seed=${lineage_seed}/generation=${generation}/agent_id=${agent_id}" }

    input:
    // Accept all URIs as val() strings (no file staging - files accessed via cloud URIs)
    // Hashes are included for cache invalidation when content changes (not used in script)
    tuple val(config_uri), val(config_hash), val(sim_data_uri), val(sim_data_hash), val(variant), val(lineage_seed), val(generation), val(sim_seed), val(initial_state_uri), val(agent_id), val(prev_division_time)

    output:
    // Pass daughter state URIs as val() strings (no file staging)
    tuple val(config_uri), val(config_hash), val(sim_data_uri), val(sim_data_hash), val(variant), val(lineage_seed), val(next_generation), val(seed_d0), env('daughter_state_0_uri'), val(agent_id_d0), env('division_time'), emit: nextGen0
    tuple val(config_uri), val(config_hash), val(sim_data_uri), val(sim_data_hash), val(variant), val(lineage_seed), val(next_generation), val(seed_d1), env('daughter_state_1_uri'), val(agent_id_d1), env('division_time'), emit: nextGen1
    // sim_data_hash is passed through for cache invalidation in downstream analysis processes
    tuple val(sim_data_uri), val(sim_data_hash), val(params.experimentId), val(variant), val(lineage_seed), val(generation), val(agent_id), emit: metadata

    script:
    next_generation = generation + 1
    agent_id_d0 = agent_id + '0'
    agent_id_d1 = agent_id + '1'
    seed_d0 = sim_seed + 1
    seed_d1 = sim_seed + 2
    def daughter_outdir = "${params.publishDir}/${params.experimentId}/daughter_states/variant=${variant}/seed=${lineage_seed}/generation=${generation}/agent_id=${agent_id}"
    """
    # Belt-and-suspenders: source /vEcoli/.env so ECOLI_SOURCES resolves
    # correctly even when the container's ENTRYPOINT is bypassed.
    [[ -f /vEcoli/.env ]] && set -a && source /vEcoli/.env && set +a || true
    # Use 1 Polars thread to avoid oversubscription on HPC/cloud
    # Access config, sim_data, and initial_state directly via cloud URIs (fsspec handles both S3 and GCS)
    # Daughter states are written directly to cloud storage via fsspec
    PYTHONUNBUFFERED=1 POLARS_MAX_THREADS=1 python ${params.projectRoot}/ecoli/experiments/ecoli_master_sim.py \\
        --config "${config_uri}" \\
        --sim_data_path "${sim_data_uri}" \\
        --initial_state_file "${initial_state_uri}" \\
        --daughter_outdir "${daughter_outdir}" \\
        --variant ${variant} \\
        --seed ${sim_seed} \\
        --lineage_seed ${lineage_seed} \\
        --agent_id \'${agent_id}\' \\
        --initial_global_time ${prev_division_time}
    # Set division_time environment variable from file written by ecoli_master_sim.py
    source division_time.sh
    # Read daughter state URIs from files written by ecoli_master_sim.py
    export daughter_state_0_uri=\$(cat daughter_state_0_uri.txt)
    export daughter_state_1_uri=\$(cat daughter_state_1_uri.txt)
    """

    stub:
    next_generation = generation + 1
    agent_id_d0 = agent_id + '0'
    agent_id_d1 = agent_id + '1'
    seed_d0 = sim_seed + 1
    seed_d1 = sim_seed + 2
    def daughter_outdir = "${params.publishDir}/${params.experimentId}/daughter_states/variant=${variant}/seed=${lineage_seed}/generation=${generation}/agent_id=${agent_id}"
    """
    mkdir -p ${daughter_outdir}
    echo "$config_uri $sim_data_uri $lineage_seed $generation" > ${daughter_outdir}/daughter_state_0.json
    echo "$initial_state_uri $sim_seed" > ${daughter_outdir}/daughter_state_1.json
    export daughter_state_0_uri=${daughter_outdir}/daughter_state_0.json
    export daughter_state_1_uri=${daughter_outdir}/daughter_state_1.json
    export division_time=1000
    """
}
