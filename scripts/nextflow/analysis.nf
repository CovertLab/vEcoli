process analysisSingle {
    publishDir "${params.publishDir}/${params.experimentId}/analyses/variant=${variant}/seed=${seed}/generation=${generation}/agent_id=${agent_id}"

    errorStrategy { (task.attempt <= maxRetries) ? 'retry' : 'ignore' }

    input:
    path config
    path kb
    tuple path(sim_data), val(experiment_id), val(variant), val(seed), val(generation), val(agent_id)

    output:
    path '*'

    script:
    """
    PYTHONPATH=${params.projectRoot} python ${params.projectRoot}/scripts/run_analysis.py -c $config \\
        --sim-data-path \'$sim_data\' \\
        --validation-data-path \'$kb/validationData.cPickle\' \\
        --experiment_id \'$experiment_id\' \\
        --variant $variant \\
        --seed $seed \\
        --generation $generation \\
        --agent_id \'$agent_id\' \\
        -o \$(pwd)
    """

    stub:
    """
    mkdir plots
    echo -e "\'$sim_data\'\n\n\'$kb/validationData.cPickle\'\n\n\'$experiment_id\'\n\n\'$variant\'\n\n$seed\n\n$generation\n\n$agent_id" > plots/test.txt
    """
}

process analysisMulticell {
    publishDir "${params.publishDir}/${params.experimentId}/analyses/variant=${variant}/seed=${seed}/generation=${generation}/agent_id=${agent_id}"

    errorStrategy { (task.attempt <= maxRetries) ? 'retry' : 'ignore' }

    input:
    path config
    path kb
    tuple path(sim_data), val(experiment_id), val(variant), val(seed), val(generation), val(agent_id)

    output:
    path '*'

    script:
    """
    PYTHONPATH=${params.projectRoot} python ${params.projectRoot}/scripts/run_analysis.py -c $config \\
        --sim-data-path \'${sim_data.join("\' \'")}\' \\
        --validation-data-path \'$kb/validationData.cPickle\' \\
        --experiment_id \'${experiment_id.join("\' \'")}\' \\
        --variant $variant \\
        --seed $seed \\
        --generation $generation \\
        -o \$(pwd)
    """

    stub:
    """
    mkdir plots
    echo -e "\'${sim_data.join("\' \'")}\'\n\n\'${experiment_id.join("\' \'")}\'\n\n\'$variant\'\n\n$seed\n\n$generation" > plots/test.txt
    """
}

process analysisMultigeneration {
    publishDir "${params.publishDir}/${params.experimentId}/analyses/variant=${variant}/seed=${seed}"

    input:
    path config
    path kb
    tuple path(sim_data), val(experiment_id), val(variant), val(seed), val(generation), val(agent_id)

    output:
    path '*'

    script:
    """
    PYTHONPATH=${params.projectRoot} python ${params.projectRoot}/scripts/run_analysis.py -c $config \\
        --sim-data-path \'${sim_data.join("\' \'")}\' \\
        --validation-data-path \'$kb/validationData.cPickle\' \\
        --experiment_id \'${experiment_id.join("\' \'")}\' \\
        --variant $variant \\
        --seed $seed \\
        -o \$(pwd)
    """

    stub:
    """
    mkdir plots
    echo -e "\'${sim_data.join("\' \'")}\'\n\n\'${experiment_id.join("\' \'")}\'\n\n\'$variant\'\n\n$seed\n\n" > plots/test.txt
    """
}

process analysisMultiseed {
    publishDir "${params.publishDir}/${params.experimentId}/analyses/variant=${variant}"

    input:
    path config
    path kb
    tuple path(sim_data), val(experiment_id), val(variant), val(seed), val(generation), val(agent_id)

    output:
    path '*'

    script:
    """
    PYTHONPATH=${params.projectRoot} python ${params.projectRoot}/scripts/run_analysis.py -c $config \\
        --sim-data-path \'${sim_data.join("\' \'")}\' \\
        --validation-data-path \'$kb/validationData.cPickle\' \\
        --experiment_id \'${experiment_id.join("\' \'")}\' \\
        --variant $variant \\
        -o \$(pwd)
    """

    stub:
    """
    mkdir plots
    echo -e "\'${sim_data.join("\' \'")}\'\n\n\'${experiment_id.join("\' \'")}\'\n\n\'$variant\'" > plots/test.txt
    """
}

process analysisMultivariant {
    publishDir "${params.publishDir}/${params.experimentId}/analyses"

    input:
    path config
    path kb
    tuple path(sim_data), val(experiment_id), val(variant), val(seed), val(generation), val(agent_id)

    output:
    path '*'

    script:
    """
    PYTHONPATH=${params.projectRoot} python ${params.projectRoot}/scripts/run_analysis.py -c $config \\
        --sim-data-path \'${sim_data.join("\' \'")}\' \\
        --validation-data-path \'$kb/validationData.cPickle\' \\
        --experiment_id \'$experiment_id\' \\
        --variant ${variant.join("\' \'")} \\
        -o \$(pwd)
    """

    stub:
    """
    mkdir plots
    echo -e "\'${sim_data.join("\' \'")}\'\n\n\'$experiment_id\'\n\n\'${variant.join("\' \'")}\'" > plots/test.txt
    """
}
