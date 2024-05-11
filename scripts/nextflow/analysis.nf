process analysisSingle {
    publishDir "${params.publishDir}/analyses/${variant}/${seed}/${generation}/${cell_id}"

    input:
    path config
    path kb
    tuple path(sim_data, stageAs: 'variant_sim_data_*.cPickle'), val(experiment_id), val(variant), val(seed), val(generation), val(cell_id)

    output:
    path 'plots/*'

    script:
    """
    python ${params.projectRoot}/scripts/run_analysis.py -c $config \
        --sim-data-path $sim_data \
        --validation-data-path $kb/validationData.cPickle \
        --experiment_id $experiment_id \
        --variant $variant \
        --seed $seed \
        --generation $generation \ 
        --cell_id $cell_id \
        -o \$(pwd)
    """

    stub:
    """
    mkdir plots
    echo -e "$sim_data\n\n$seed\n\n$generation\n\n$cell_id\
        \n\n$kb\n\n$config" > plots/test.txt
    """
}

process analysisMulticell {
    publishDir "${params.publishDir}/analyses/${variant}/${seed}/${generation}/${cell_id}"

    input:
    path config
    path kb
    tuple path(sim_data, stageAs: 'variant_sim_data_*.cPickle'), val(experiment_id), val(variant), val(seed), val(generation), val(cell_id)

    output:
    path 'plots/*'

    script:
    """
    python ${params.projectRoot}/scripts/run_analysis.py -c $config \
        --sim-data-path $sim_data \
        --validation-data-path $kb/validationData.cPickle \
        --experiment_id $experiment_id \
        --variant $variant \
        --seed $seed \
        --generation $generation \ 
        -o \$(pwd)
    """

    stub:
    """
    mkdir plots
    echo -e "$sim_data\n\n$seed\n\n$generation\n\n$cell_id\
        \n\n$kb\n\n$config" > plots/test.txt
    """
}

process analysisMultigeneration {
    publishDir "${params.publishDir}/analyses/${variant}/${seed}"

    input:
    path config
    path kb
    tuple path(sim_data, stageAs: 'variant_sim_data_*.cPickle'), val(experiment_id), val(variant), val(seed), val(generation), val(cell_id)

    output:
    path 'plots/*'

    script:
    """
    python ${params.projectRoot}/scripts/run_analysis.py -c $config \
        --sim-data-path $sim_data \
        --validation-data-path $kb/validationData.cPickle \
        --experiment_id $experiment_id \
        --variant $variant \
        --seed $seed \
        -o \$(pwd)
    """

    stub:
    """
    mkdir plots
    echo -e "$sim_data\n\n$seed\n\n$generation\n\n$cell_id\
        \n\n$kb\n\n$config" > plots/test.txt
    """
}

process analysisMultiseed {
    publishDir "${params.publishDir}/analyses/${variant}"

    input:
    path config
    path kb
    tuple path(sim_data, stageAs: 'variant_sim_data_*.cPickle'), val(experiment_id), val(variant), val(seed), val(generation), val(cell_id)

    output:
    path 'plots/*'

    script:
    """
    python ${params.projectRoot}/scripts/run_analysis.py -c $config \
        --sim-data-path $sim_data \
        --validation-data-path $kb/validationData.cPickle \
        --experiment_id $experiment_id \
        --variant $variant \
        -o \$(pwd)
    """

    stub:
    """
    mkdir plots
    echo -e "$sim_data\n\n$seed\n\n$generation\n\n$cell_id\
        \n\n$kb\n\n$config" > plots/test.txt
    """
}

process analysisMultivariant {
    publishDir "${params.publishDir}/analyses"

    input:
    path config
    path kb
    tuple path(sim_data, stageAs: 'variant_sim_data_*.cPickle'), val(experiment_id), val(variant), val(seed), val(generation), val(cell_id)

    output:
    path 'plots/*'

    script:
    """
    python ${params.projectRoot}/scripts/run_analysis.py -c $config \
        --sim-data-path ${sim_data.join(" ")} \
        --validation-data-path $kb/validationData.cPickle \
        --experiment_id $experiment_id \
        --variant ${variant.join(" ")}
        -o \$(pwd)
    """

    stub:
    """
    mkdir plots
    echo -e "$sim_data\n\n$seed\n\n$generation\n\n$cell_id\
        \n\n$kb\n\n$config" > plots/test.txt
    """
}
