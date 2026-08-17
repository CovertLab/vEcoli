process analysisSingle {
    publishDir { "${params.publishDir}/${params.experimentId}/analyses/variant=${variant}/lineage_seed=${lineage_seed}/generation=${generation}/agent_id=${agent_id}" }, mode: "copy"

    tag { "variant=${variant}/lineage_seed=${lineage_seed}/generation=${generation}/agent_id=${agent_id}" }

    label "analysis"

    input:
    // Accept URIs and hashes for cache invalidation
    tuple val(config_uri), val(config_hash), val(kb_uri), val(kb_hash)
    // Accept sim_data_uri as val() string to avoid staging issues with cloud URIs
    // sim_data_hash is included for cache invalidation when content changes
    tuple val(sim_data_uri), val(sim_data_hash), val(experiment_id), val(variant), val(lineage_seed), val(generation), val(agent_id)
    val variant_metadata_uri

    output:
    path 'plots/analysis=*/*'
    path 'plots/analysis=*/metadata.json'

    script:
    """
    mkdir -p plots
    PYTHONUNBUFFERED=1 python ${params.projectRoot}/runscripts/analysis.py --config "${config_uri}" \\
        --sim_data_path "${sim_data_uri}" \\
        --validation_data_path "${kb_uri}/validationData.cPickle" \\
        --experiment_id "$experiment_id" \\
        --variant $variant \\
        --lineage_seed $lineage_seed \\
        --generation $generation \\
        --agent_id "$agent_id" \\
        --variant_metadata_path "${variant_metadata_uri}" \\
        --cpus ${params.duckdb_threads} \\
        -o "\$(pwd)/plots" \\
        -t single
    cd plots
    cd experiment_id=*
    cd variant=*
    cd lineage_seed=*
    cd generation=*
    cd agent_id=*
    mv analysis=* ../../../../..
    cd ../../../../..
    rm -rf experiment_id=*
    """

    stub:
    """
    mkdir -p plots/analysis=stub
    echo -e "Single\n\n$sim_data_uri\n\n$kb_uri/validationData.cPickle\n\n$experiment_id\n\n$variant\n\n$lineage_seed\n\n$generation\n\n$agent_id" > plots/analysis=stub/test.txt
    touch plots/analysis=stub/metadata.json
    """
}

process analysisMultiDaughter {
    publishDir { "${params.publishDir}/${params.experimentId}/analyses/variant=${variant}/lineage_seed=${lineage_seed}/generation=${generation}" }, mode: "copy"

    tag { "variant=${variant}/lineage_seed=${lineage_seed}/generation=${generation}/analysis=${analysis_name}" }

    label "analysis"

    input:
    tuple val(config_uri), val(config_hash), val(kb_uri), val(kb_hash)
    // group_size is included for cache invalidation when # of sims included
    // in this analysis run changes (e.g. when resuming workflow with --resume)
    tuple val(sim_data_uri), val(sim_data_hash), val(experiment_id), val(variant), val(lineage_seed), val(generation), val(group_size), val(analysis_name)
    val variant_metadata_uri

    output:
    path 'plots/analysis=*/*'
    path 'plots/analysis=*/metadata.json'

    script:
    """
    mkdir -p plots
    PYTHONUNBUFFERED=1 python ${params.projectRoot}/runscripts/analysis.py --config "${config_uri}" \\
        --sim_data_path "${sim_data_uri}" \\
        --validation_data_path "${kb_uri}/validationData.cPickle" \\
        --experiment_id "$experiment_id" \\
        --variant $variant \\
        --lineage_seed $lineage_seed \\
        --generation $generation \\
        --variant_metadata_path "${variant_metadata_uri}" \\
        --analysis_name "$analysis_name" \\
        --cpus ${params.duckdb_threads} \\
        -o "\$(pwd)/plots" \\
        -t multidaughter
    cd plots
    cd experiment_id=*
    cd variant=*
    cd lineage_seed=*
    cd generation=*
    mv analysis=* ../../../..
    cd ../../../..
    rm -rf experiment_id=*
    """

    stub:
    """
    mkdir -p plots/analysis=$analysis_name
    echo -e "Multicell\n\n$sim_data_uri\n\n$experiment_id\n\n$variant\n\n$lineage_seed\n\n$generation\n\ngroup_size=$group_size\n\n$analysis_name" > plots/analysis=$analysis_name/test.txt
    touch plots/analysis=$analysis_name/metadata.json
    """
}

process analysisMultiGeneration {
    publishDir { "${params.publishDir}/${params.experimentId}/analyses/variant=${variant}/lineage_seed=${lineage_seed}" }, mode: "copy"

    tag { "variant=${variant}/lineage_seed=${lineage_seed}/analysis=${analysis_name}" }

    label "analysis"

    input:
    tuple val(config_uri), val(config_hash), val(kb_uri), val(kb_hash)
    tuple val(sim_data_uri), val(sim_data_hash), val(experiment_id), val(variant), val(lineage_seed), val(group_size), val(analysis_name)
    val variant_metadata_uri

    output:
    path 'plots/analysis=*/*'
    path 'plots/analysis=*/metadata.json'

    script:
    """
    mkdir -p plots
    PYTHONUNBUFFERED=1 python ${params.projectRoot}/runscripts/analysis.py --config "${config_uri}" \\
        --sim_data_path "${sim_data_uri}" \\
        --validation_data_path "${kb_uri}/validationData.cPickle" \\
        --experiment_id "$experiment_id" \\
        --variant $variant \\
        --lineage_seed $lineage_seed \\
        --variant_metadata_path "${variant_metadata_uri}" \\
        --analysis_name "$analysis_name" \\
        --cpus ${params.duckdb_threads} \\
        -o "\$(pwd)/plots" \\
        -t multigeneration
    cd plots
    cd experiment_id=*
    cd variant=*
    cd lineage_seed=*
    mv analysis=* ../../..
    cd ../../..
    rm -rf experiment_id=*
    """

    stub:
    """
    mkdir -p plots/analysis=$analysis_name
    echo -e "Multigeneration\n\n$sim_data_uri\n\n$experiment_id\n\n$variant\n\n$lineage_seed\n\ngroup_size=$group_size\n\n$analysis_name" > plots/analysis=$analysis_name/test.txt
    touch plots/analysis=$analysis_name/metadata.json
    """
}

process analysisMultiSeed {
    publishDir { "${params.publishDir}/${params.experimentId}/analyses/variant=${variant}" }, mode: "copy"

    tag { "variant=${variant}/analysis=${analysis_name}" }

    label "analysis"

    input:
    tuple val(config_uri), val(config_hash), val(kb_uri), val(kb_hash)
    tuple val(sim_data_uri), val(sim_data_hash), val(experiment_id), val(variant), val(group_size), val(analysis_name)
    val variant_metadata_uri

    output:
    path 'plots/analysis=*/*'
    path 'plots/analysis=*/metadata.json'

    script:
    """
    mkdir -p plots
    PYTHONUNBUFFERED=1 python ${params.projectRoot}/runscripts/analysis.py --config "${config_uri}" \\
        --sim_data_path "${sim_data_uri}" \\
        --validation_data_path "${kb_uri}/validationData.cPickle" \\
        --experiment_id "$experiment_id" \\
        --variant $variant \\
        --variant_metadata_path "${variant_metadata_uri}" \\
        --analysis_name "$analysis_name" \\
        --cpus ${params.duckdb_threads} \\
        -o "\$(pwd)/plots" \\
        -t multiseed
    cd plots
    cd experiment_id=*
    cd variant=*
    mv analysis=* ../..
    cd ../..
    rm -rf experiment_id=*
    """

    stub:
    """
    mkdir -p plots/analysis=$analysis_name
    echo -e "Multiseed\n\n$sim_data_uri\n\n$experiment_id\n\n$variant\n\ngroup_size=$group_size\n\n$analysis_name" > plots/analysis=$analysis_name/test.txt
    touch plots/analysis=$analysis_name/metadata.json
    """
}

process analysisMultiVariant {
    publishDir { "${params.publishDir}/${params.experimentId}/analyses" }, mode: "copy"

    label "analysis"

    input:
    tuple val(config_uri), val(config_hash), val(kb_uri), val(kb_hash)
    tuple val(sim_data_uris), val(sim_data_hashes), val(experiment_id), val(variant), val(group_size), val(analysis_name)
    val variant_metadata_uri

    output:
    path 'plots/analysis=*/*'
    path 'plots/analysis=*/metadata.json'

    script:
    """
    mkdir -p plots
    PYTHONUNBUFFERED=1 python ${params.projectRoot}/runscripts/analysis.py --config "${config_uri}" \\
        --sim_data_path "${sim_data_uris.join('" "')}" \\
        --validation_data_path "${kb_uri}/validationData.cPickle" \\
        --experiment_id "$experiment_id" \\
        --variant ${variant.join(" ")} \\
        --variant_metadata_path "${variant_metadata_uri}" \\
        --analysis_name "$analysis_name" \\
        --cpus ${params.duckdb_threads} \\
        -o "\$(pwd)/plots" \\
        -t multivariant
    cd plots
    cd experiment_id=*
    mv analysis=* ..
    cd ..
    rm -rf experiment_id=*
    """

    stub:
    """
    mkdir -p plots/analysis=$analysis_name
    echo -e "Multivariant\n\n${sim_data_uris.join('" "')}\n\n$experiment_id\n\n${variant.join(' ')}\n\ngroup_size=$group_size\n\n$analysis_name" > plots/analysis=$analysis_name/test.txt
    touch plots/analysis=$analysis_name/metadata.json
    """
}
