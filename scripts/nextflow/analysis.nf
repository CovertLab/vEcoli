process analysisSingle {
    publishDir "${params.publishDir}/${params.experimentId}/analyses/variant=${variant}/lineage_seed=${lineage_seed}/generation=${generation}/agent_id=${agent_id}"

    errorStrategy { (task.exitStatus in [137, 140, 143]) && (task.attempt <= maxRetries) ? 'retry' : 'terminate' }

    tag "variant=${variant}/lineage_seed=${lineage_seed}/generation=${generation}/agent_id=${agent_id}"

    input:
    path config
    path kb
    tuple path(sim_data), val(experiment_id), val(variant), val(lineage_seed), val(generation), val(agent_id)

    output:
    path '*'

    script:
    """
    PYTHONPATH=${params.projectRoot} python ${params.projectRoot}/scripts/run_analysis.py -c $config \\
        --sim-data-path "$sim_data" \\
        --validation-data-path "$kb/validationData.cPickle" \\
        --experiment_id "$experiment_id" \\
        --variant $variant \\
        --lineage_seed $lineage_seed \\
        --generation $generation \\
        --agent_id "$agent_id" \\
        -o \$(pwd)
    """

    stub:
    """
    mkdir -p plots
    echo -e "Single\n\n"$sim_data"\n\n"$kb/validationData.cPickle"\n\n"$experiment_id"\n\n"$variant"\n\n$lineage_seed\n\n$generation\n\n$agent_id" > plots/test.txt
    """
}

process analysisMultiDaughter {
    publishDir "${params.publishDir}/${params.experimentId}/analyses/variant=${variant}/lineage_seed=${lineage_seed}/generation=${generation}"

    errorStrategy { (task.exitStatus in [137, 140, 143]) && (task.attempt <= maxRetries) ? 'retry' : 'terminate' }

    tag "variant=${variant}/lineage_seed=${lineage_seed}/generation=${generation}"

    input:
    path config
    path kb
    tuple path(sim_data), val(experiment_id), val(variant), val(lineage_seed), val(generation)

    output:
    path '*'

    script:
    """
    PYTHONPATH=${params.projectRoot} python ${params.projectRoot}/scripts/run_analysis.py -c $config \\
        --sim-data-path "$sim_data" \\
        --validation-data-path "$kb/validationData.cPickle" \\
        --experiment_id "$experiment_id" \\
        --variant $variant \\
        --lineage_seed $lineage_seed \\
        --generation $generation \\
        -o \$(pwd)
    """

    stub:
    """
    mkdir -p plots
    echo -e "Multicell\n\n"$sim_data"\n\n"$experiment_id"\n\n"$variant"\n\n$lineage_seed\n\n$generation" > plots/test.txt
    """
}

process analysisMultiGeneration {
    publishDir "${params.publishDir}/${params.experimentId}/analyses/variant=${variant}/lineage_seed=${lineage_seed}"

    errorStrategy { (task.exitStatus in [137, 140, 143]) && (task.attempt <= maxRetries) ? 'retry' : 'terminate' }

    tag "variant=${variant}/lineage_seed=${lineage_seed}"

    input:
    path config
    path kb
    tuple path(sim_data), val(experiment_id), val(variant), val(lineage_seed)

    output:
    path '*'

    script:
    """
    PYTHONPATH=${params.projectRoot} python ${params.projectRoot}/scripts/run_analysis.py -c $config \\
        --sim-data-path "$sim_data" \\
        --validation-data-path "$kb/validationData.cPickle" \\
        --experiment_id "$experiment_id" \\
        --variant $variant \\
        --lineage_seed $lineage_seed \\
        -o \$(pwd)
    """

    stub:
    """
    mkdir -p plots
    echo -e "Multigeneration\n\n"$sim_data"\n\n"$experiment_id"\n\n"$variant"\n\n$lineage_seed\n\n" > plots/test.txt
    """
}

process analysisMultiSeed {
    publishDir "${params.publishDir}/${params.experimentId}/analyses/variant=${variant}"

    errorStrategy { (task.exitStatus in [137, 140, 143]) && (task.attempt <= maxRetries) ? 'retry' : 'terminate' }

    tag "variant=${variant}"

    input:
    path config
    path kb
    tuple path(sim_data), val(experiment_id), val(variant)

    output:
    path '*'

    script:
    """
    PYTHONPATH=${params.projectRoot} python ${params.projectRoot}/scripts/run_analysis.py -c $config \\
        --sim-data-path "$sim_data" \\
        --validation-data-path "$kb/validationData.cPickle" \\
        --experiment_id "$experiment_id" \\
        --variant $variant \\
        -o \$(pwd)
    """

    stub:
    """
    mkdir -p plots
    echo -e "Multiseed\n\n"$sim_data"\n\n"$experiment_id"\n\n"$variant"" > plots/test.txt
    """
}

process analysisMultiVariant {
    publishDir "${params.publishDir}/${params.experimentId}/analyses"

    errorStrategy { (task.exitStatus in [137, 140, 143]) && (task.attempt <= maxRetries) ? 'retry' : 'terminate' }

    input:
    path config
    path kb
    tuple path(sim_data, stageAs: 'simData*.cPickle'), val(experiment_id), val(variant)

    output:
    path '*'

    script:
    """
    PYTHONPATH=${params.projectRoot} python ${params.projectRoot}/scripts/run_analysis.py -c $config \\
        --sim-data-path "${sim_data.join("\" \"")}" \\
        --validation-data-path "$kb/validationData.cPickle" \\
        --experiment_id "$experiment_id" \\
        --variant ${variant.join("\" \"")} \\
        -o \$(pwd)
    """

    stub:
    """
    mkdir -p plots
    echo -e "Multivariant\n\n"${sim_data.join("\" \"")}"\n\n"$experiment_id"\n\n"${variant.join("\" \"")}"" > plots/test.txt
    """
}
