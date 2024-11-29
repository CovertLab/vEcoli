process runParca {
    // Run ParCa using parca_options from config JSON
    publishDir "${params.publishDir}/${params.experimentId}/parca", mode: "copy"

    cpus PARCA_CPUS

    input:
    path config

    output:
    path 'kb'

    script:
    """
    PYTHONPATH=${params.projectRoot} python ${params.projectRoot}/runscripts/parca.py --config "$config" -o "\$(pwd)"
    """

    stub:
    """
    mkdir kb
    echo "Mock sim_data" > kb/simData.cPickle
    echo "Mock raw_data" > kb/rawData.cPickle
    echo "Mock raw_validation_data" > kb/rawValidationData.cPickle
    echo "Mock validation_data" > kb/validationData.cPickle
    """
}

process analysisParca {
    publishDir "${params.publishDir}/${params.experimentId}/parca/analysis", mode: "move"

    input:
    path config
    path kb

    output:
    path 'plots/*'

    script:
    """
    PYTHONPATH=${params.projectRoot} python ${params.projectRoot}/runscripts/analysis.py --config "$config" \
        --sim_data_path="$kb/simData.cPickle" \
        --validation_data_path="$kb/validationData.cPickle" \
        -o "\$(pwd)/plots" \
        -t parca
    """

    stub:
    """
    mkdir plots
    echo -e "$config\n\n$kb" > plots/test.txt
    """
}

process createVariants {
    // Parse variants in config JSON to generate variants
    publishDir "${params.publishDir}/${params.experimentId}/variant_sim_data", mode: "copy"

    input:
    path config
    path kb

    output:
    path '*.cPickle', emit: variantSimData
    path 'metadata.json', emit: variantMetadata

    script:
    """
    PYTHONPATH=${params.projectRoot} python ${params.projectRoot}/runscripts/create_variants.py \
        --config "$config" --kb "$kb" -o "\$(pwd)"
    """

    stub:
    """
    cp $kb/simData.cPickle variant_1.cPickle
    echo "Mock variant 1" >> variant_1.cPickle
    cp $kb/simData.cPickle variant_2.cPickle
    echo "Mock variant 2" >> variant_2.cPickle
    echo "Mock metadata.json" > metadata.json
    cp $kb/simData.cPickle baseline.cPickle
    """
}

IMPORTS

workflow {
RUN_PARCA
    createVariants(params.config, kb)
        .variantSimData
        .flatten()
        .set { variantCh }
    createVariants.out
        .variantMetadata
        .set { variantMetadataCh }
WORKFLOW
}
