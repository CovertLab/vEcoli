process analysisSingle {
    publishDir "${params.publishDir}/${params.experimentId}/analyses/variant=${variant}/lineage_seed=${lineage_seed}/generation=${generation}/agent_id=${agent_id}", mode: "copy"

    tag "variant=${variant}/lineage_seed=${lineage_seed}/generation=${generation}/agent_id=${agent_id}"

    label "slurm_submit"

    input:
    path config
    path kb
    tuple path(sim_data), val(experiment_id), val(variant), val(lineage_seed), val(generation), val(agent_id)
    path variant_metadata

    output:
    path 'plots/*'
    path 'plots/metadata.json'

    script:
    """
    mkdir -p plots
    python ${params.projectRoot}/runscripts/analysis.py --config $config \\
        --sim_data_path "$sim_data" \\
        --validation_data_path "$kb/validationData.cPickle" \\
        --experiment_id "$experiment_id" \\
        --variant $variant \\
        --lineage_seed $lineage_seed \\
        --generation $generation \\
        --agent_id "$agent_id" \\
        --variant_metadata_path "${variant_metadata}" \\
        -o "\$(pwd)/plots" \\
        -t single
    cd plots
    cd experiment_id=*
    cd variant=*
    cd lineage_seed=*
    cd generation=*
    cd agent_id=*
    mv * ../../../../..
    cd ../../../../..
    rm -rf experiment_id=*
    """

    stub:
    """
    mkdir -p plots
    echo -e "Single\n\n"$sim_data"\n\n"$kb/validationData.cPickle"\n\n"$experiment_id"\n\n"$variant"\n\n$lineage_seed\n\n$generation\n\n$agent_id" > plots/test.txt
    touch plots/metadata.json
    """
}

process analysisMultiDaughter {
    publishDir "${params.publishDir}/${params.experimentId}/analyses/variant=${variant}/lineage_seed=${lineage_seed}/generation=${generation}", mode: "copy"

    tag "variant=${variant}/lineage_seed=${lineage_seed}/generation=${generation}"

    label "slurm_submit"

    input:
    path config
    path kb
    tuple path(sim_data), val(experiment_id), val(variant), val(lineage_seed), val(generation)
    path variant_metadata

    output:
    path 'plots/*'
    path 'plots/metadata.json'

    script:
    """
    mkdir -p plots
    python ${params.projectRoot}/runscripts/analysis.py --config $config \\
        --sim_data_path "$sim_data" \\
        --validation_data_path "$kb/validationData.cPickle" \\
        --experiment_id "$experiment_id" \\
        --variant $variant \\
        --lineage_seed $lineage_seed \\
        --generation $generation \\
        --variant_metadata_path "${variant_metadata}" \\
        -o "\$(pwd)/plots" \\
        -t multidaughter
    cd plots
    cd experiment_id=*
    cd variant=*
    cd lineage_seed=*
    cd generation=*
    mv * ../../../..
    cd ../../../..
    rm -rf experiment_id=*
    """

    stub:
    """
    mkdir -p plots
    echo -e "Multicell\n\n"$sim_data"\n\n"$experiment_id"\n\n"$variant"\n\n$lineage_seed\n\n$generation" > plots/test.txt
    touch plots/metadata.json
    """
}

process analysisMultiGeneration {
    publishDir "${params.publishDir}/${params.experimentId}/analyses/variant=${variant}/lineage_seed=${lineage_seed}", mode: "copy"

    tag "variant=${variant}/lineage_seed=${lineage_seed}"

    label "slurm_submit"

    input:
    path config
    path kb
    tuple path(sim_data), val(experiment_id), val(variant), val(lineage_seed)
    path variant_metadata

    output:
    path 'plots/*'
    path 'plots/metadata.json'

    script:
    """
    mkdir -p plots
    python ${params.projectRoot}/runscripts/analysis.py --config $config \\
        --sim_data_path "$sim_data" \\
        --validation_data_path "$kb/validationData.cPickle" \\
        --experiment_id "$experiment_id" \\
        --variant $variant \\
        --lineage_seed $lineage_seed \\
        --variant_metadata_path "${variant_metadata}" \\
        -o "\$(pwd)/plots" \\
        -t multigeneration
    cd plots
    cd experiment_id=*
    cd variant=*
    cd lineage_seed=*
    mv * ../../..
    cd ../../..
    rm -rf experiment_id=*
    """

    stub:
    """
    mkdir -p plots
    echo -e "Multigeneration\n\n"$sim_data"\n\n"$experiment_id"\n\n"$variant"\n\n$lineage_seed\n\n" > plots/test.txt
    touch plots/metadata.json
    """
}

process analysisMultiSeed {
    publishDir "${params.publishDir}/${params.experimentId}/analyses/variant=${variant}", mode: "copy"

    tag "variant=${variant}"

    label "slurm_submit"

    input:
    path config
    path kb
    tuple path(sim_data), val(experiment_id), val(variant)
    path variant_metadata

    output:
    path 'plots/*'
    path 'plots/metadata.json'

    script:
    """
    mkdir -p plots
    python ${params.projectRoot}/runscripts/analysis.py --config $config \\
        --sim_data_path "$sim_data" \\
        --validation_data_path "$kb/validationData.cPickle" \\
        --experiment_id "$experiment_id" \\
        --variant $variant \\
        --variant_metadata_path "${variant_metadata}" \\
        -o "\$(pwd)/plots" \\
        -t multiseed
    cd plots
    cd experiment_id=*
    cd variant=*
    mv * ../..
    cd ../..
    rm -rf experiment_id=*
    """

    stub:
    """
    mkdir -p plots
    echo -e "Multiseed\n\n"$sim_data"\n\n"$experiment_id"\n\n"$variant"" > plots/test.txt
    touch plots/metadata.json
    """
}

process analysisMultiVariant {
    publishDir "${params.publishDir}/${params.experimentId}/analyses", mode: "copy"

    label "slurm_submit"

    input:
    path config
    path kb
    tuple path(sim_data, stageAs: 'simData*.cPickle'), val(experiment_id), val(variant)
    path variant_metadata

    output:
    path 'plots/*'
    path 'plots/metadata.json'

    script:
    """
    mkdir -p plots
    python ${params.projectRoot}/runscripts/analysis.py --config $config \\
        --sim_data_path "${sim_data.join("\" \"")}" \\
        --validation_data_path "$kb/validationData.cPickle" \\
        --experiment_id "$experiment_id" \\
        --variant ${variant.join(" ")} \\
        --variant_metadata_path "${variant_metadata}" \\
        -o "\$(pwd)/plots" \\
        -t multivariant
    cd plots
    cd experiment_id=*
    mv * ..
    cd ..
    rm -rf experiment_id=*
    """

    stub:
    """
    mkdir -p plots
    echo -e "Multivariant\n\n"${sim_data.join("\" \"")}"\n\n"$experiment_id"\n\n"${variant.join("\" \"")}"" > plots/test.txt
    touch plots/metadata.json
    """
}
