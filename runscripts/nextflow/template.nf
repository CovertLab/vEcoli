process runParca {
    // Run ParCa using parca_options from config JSON
    publishDir "${params.publishDir}/${params.experimentId}/parca", mode: "copy"

    label "parca"

    input:
    path config

    output:
    path 'kb'

    script:
    """
    python ${params.projectRoot}/runscripts/parca.py --config "$config" -o "\$(pwd)"
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
    publishDir "${params.publishDir}/${params.experimentId}/parca/analysis", mode: "copy"

    label "slurm_submit"

    input:
    path config
    path kb

    output:
    path 'plots/*'

    script:
    """
    python ${params.projectRoot}/runscripts/analysis.py --config "$config" \
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

    label "slurm_submit"

    input:
    path config
    path kb

    output:
    path '*.cPickle', emit: variantSimData
    path 'metadata.json', emit: variantMetadata

    script:
    """
    python ${params.projectRoot}/runscripts/create_variants.py \
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

process hqWorker {
    cpus { num_sims }

    memory {
        if ( task.exitStatus in [137, 140] ) {
            task.cpus * 4.GB + 4.GB * (task.attempt - 1)
        } else {
            task.cpus * 4.GB
        }
    }
    time 24.h
    maxRetries 10

    tag "hq_${params.experimentId}_${task.index}"

    executor 'slurm'
    queue 'owners,normal'
    // Run on newer, faster CPUs
    clusterOptions '--prefer="CPU_GEN:GEN|CPU_GEN:SPR" --constraint="CPU_GEN:RME|CPU_GEN:MLN|CPU_GEN:BGM|CPU_GEN:SIE|CPU_GEN:GEN|CPU_GEN:SPR"'
    container null

    input:
    val num_sims

    script:
    server_dir = "${params.publishDir}/${params.experimentId}/nextflow/.hq-server"
    """
    # Start HyperQueue worker with specified options
    hq worker start --manager slurm \\
        --server-dir ${server_dir} \\
        --cpus ${task.cpus} \\
        --resource "mem=sum(${task.cpus * 4096})" \\
        --idle-timeout 5m &

    worker_pid=\$!
    wait \$worker_pid
    exit_code=\$?

    # Only exit with 0 if the exit code is 0 or 1
    # This allows code 1 to be treated as success but propagates all other errors
    if [ \$exit_code -eq 0 ] || [ \$exit_code -eq 1 ]; then
        exit 0
    else
        # Forward the original error code to Nextflow
        exit \$exit_code
    fi
    """

    stub:
    """
    echo "Started HyperQueue worker for $num_sims" \\
        >> $server_dir/worker.log
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
    // Start a HyperQueue worker for every 4 concurrent sims
    if ( params.hyperqueue ) {
        variantCh.combine( seedCh )
            .buffer( size: 4, remainder: true )
            .map { it.size() }
            .set { hqChannel }
        hqWorker( hqChannel )
    }
}
