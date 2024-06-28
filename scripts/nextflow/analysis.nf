process analysisSingle {
    publishDir "${params.publishDir}/${params.experimentId}/analyses/variant=${variant}/lineage_seed=${lineage_seed}/generation=${generation}/agent_id=${agent_id}"

    tag "variant=${variant}/lineage_seed=${lineage_seed}/generation=${generation}/agent_id=${agent_id}"

    input:
    path config
    path kb
    tuple path(sim_data), val(experiment_id), val(variant), val(lineage_seed), val(generation), val(agent_id)
    path variant_metadata

    output:
    path '*'

    script:
    """
    mkdir -p plots
    PYTHONPATH=${params.projectRoot} python ${params.projectRoot}/scripts/analysis.py -c $config \\
        --sim-data-path "$sim_data" \\
        --validation-data-path "$kb/validationData.cPickle" \\
        --experiment_id "$experiment_id" \\
        --variant $variant \\
        --lineage_seed $lineage_seed \\
        --generation $generation \\
        --agent_id "$agent_id" \\
        --variant-metadata-path ${variant_metadata} \\
        -o \$(pwd)/plots
    """

    stub:
    """
    mkdir -p plots
    echo -e "Single\n\n"$sim_data"\n\n"$kb/validationData.cPickle"\n\n"$experiment_id"\n\n"$variant"\n\n$lineage_seed\n\n$generation\n\n$agent_id" > plots/test.txt
    """
}

process analysisMultiDaughter {
    publishDir "${params.publishDir}/${params.experimentId}/analyses/variant=${variant}/lineage_seed=${lineage_seed}/generation=${generation}"

    tag "variant=${variant}/lineage_seed=${lineage_seed}/generation=${generation}"

    input:
    path config
    path kb
    tuple path(sim_data), val(experiment_id), val(variant), val(lineage_seed), val(generation)
    path variant_metadata

    output:
    path '*'

    script:
    """
    mkdir -p plots
    PYTHONPATH=${params.projectRoot} python ${params.projectRoot}/scripts/analysis.py -c $config \\
        --sim-data-path "$sim_data" \\
        --validation-data-path "$kb/validationData.cPickle" \\
        --experiment_id "$experiment_id" \\
        --variant $variant \\
        --lineage_seed $lineage_seed \\
        --generation $generation \\
        --variant-metadata-path ${variant_metadata} \\
        -o \$(pwd)/plots
    """

    stub:
    """
    mkdir -p plots
    echo -e "Multicell\n\n"$sim_data"\n\n"$experiment_id"\n\n"$variant"\n\n$lineage_seed\n\n$generation" > plots/test.txt
    """
}

process analysisMultiGeneration {
    publishDir "${params.publishDir}/${params.experimentId}/analyses/variant=${variant}/lineage_seed=${lineage_seed}"

    tag "variant=${variant}/lineage_seed=${lineage_seed}"

    input:
    path config
    path kb
    tuple path(sim_data), val(experiment_id), val(variant), val(lineage_seed)
    path variant_metadata

    output:
    path '*'

    script:
    """
    mkdir -p plots
    PYTHONPATH=${params.projectRoot} python ${params.projectRoot}/scripts/analysis.py -c $config \\
        --sim-data-path "$sim_data" \\
        --validation-data-path "$kb/validationData.cPickle" \\
        --experiment_id "$experiment_id" \\
        --variant $variant \\
        --lineage_seed $lineage_seed \\
        --variant-metadata-path ${variant_metadata} \\
        -o \$(pwd)/plots
    """

    stub:
    """
    mkdir -p plots
    echo -e "Multigeneration\n\n"$sim_data"\n\n"$experiment_id"\n\n"$variant"\n\n$lineage_seed\n\n" > plots/test.txt
    """
}

process analysisMultiSeed {
    publishDir "${params.publishDir}/${params.experimentId}/analyses/variant=${variant}"

    tag "variant=${variant}"

    input:
    path config
    path kb
    tuple path(sim_data), val(experiment_id), val(variant)
    path variant_metadata

    output:
    path '*'

    script:
    """
    mkdir -p plots
    PYTHONPATH=${params.projectRoot} python ${params.projectRoot}/scripts/analysis.py -c $config \\
        --sim-data-path "$sim_data" \\
        --validation-data-path "$kb/validationData.cPickle" \\
        --experiment_id "$experiment_id" \\
        --variant $variant \\
        --variant-metadata-path ${variant_metadata} \\
        -o \$(pwd)/plots
    """

    stub:
    """
    mkdir -p plots
    echo -e "Multiseed\n\n"$sim_data"\n\n"$experiment_id"\n\n"$variant"" > plots/test.txt
    """
}

process analysisMultiVariant {
    publishDir "${params.publishDir}/${params.experimentId}/analyses"

    input:
    path config
    path kb
    tuple path(sim_data, stageAs: 'simData*.cPickle'), val(experiment_id), val(variant)
    path variant_metadata

    output:
    path '*'

    script:
    """
    mkdir -p plots
    PYTHONPATH=${params.projectRoot} python ${params.projectRoot}/scripts/analysis.py -c $config \\
        --sim-data-path "${sim_data.join("\" \"")}" \\
        --validation-data-path "$kb/validationData.cPickle" \\
        --experiment_id "$experiment_id" \\
        --variant ${variant.join(" ")} \\
        --variant-metadata-path ${variant_metadata} \\
        -o \$(pwd)/plots
    """

    stub:
    """
    mkdir -p plots
    echo -e "Multivariant\n\n"${sim_data.join("\" \"")}"\n\n"$experiment_id"\n\n"${variant.join("\" \"")}"" > plots/test.txt
    """
}
