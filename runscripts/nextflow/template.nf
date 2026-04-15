process runParca {
    // Run ParCa using parca_options from config JSON
    // Outputs directly to publishDir via fsspec and returns URIs + hash for downstream processes

    label "parca"

    input:
    path config

    output:
    // Output URIs and hash for cache invalidation in downstream processes
    // Use params.config for the full URI since config is now a staged local path
    tuple val(params.config), env('config_hash'), val("${params.publishDir}/${params.experimentId}/parca/kb"), env('kb_hash'), emit: parca_out

    script:
    def publish_path = "${params.publishDir}/${params.experimentId}/parca"
    """
    # Compute config hash for cache invalidation
    export config_hash=\$(sha256sum $config | cut -d' ' -f1)

    # Run parca, outputting directly to publish location via fsspec
    # parca.py handles cloud URIs (s3://, gs://) directly using fsspec
    PYTHONUNBUFFERED=1 python ${params.projectRoot}/runscripts/parca.py \
        --config "$config" -o "${publish_path}" --cpus ${task.cpus}

    # Read kb hash from file written by parca.py
    export kb_hash=\$(cat kb_hash.txt)
    """

    stub:
    // Only intended for local testing (not on AWS/GCP)
    def publish_path = "${params.publishDir}/${params.experimentId}/parca/kb"
    """
    export config_hash=\$(sha256sum $config | cut -d' ' -f1)
    mkdir -p ${publish_path}
    echo "Mock sim_data" > ${publish_path}/simData.cPickle
    echo "Mock raw_data" > ${publish_path}/rawData.cPickle
    echo "Mock raw_validation_data" > ${publish_path}/rawValidationData.cPickle
    echo "Mock validation_data" > ${publish_path}/validationData.cPickle
    export kb_hash=\$(sha256sum ${publish_path}/simData.cPickle | cut -d' ' -f1)
    """
}

process analysisParca {
    publishDir { "${params.publishDir}/${params.experimentId}/parca/analysis" }, mode: "copy"

    label "analysis"

    input:
    // Accept URIs and hashes (hashes for cache invalidation)
    tuple val(config_uri), val(config_hash), val(kb_uri), val(kb_hash)

    output:
    path 'plots/*'

    script:
    """
    PYTHONUNBUFFERED=1 python ${params.projectRoot}/runscripts/analysis.py \
        --config "${config_uri}" \
        --sim_data_path="${kb_uri}/simData.cPickle" \
        --validation_data_path="${kb_uri}/validationData.cPickle" \
        -o "\$(pwd)/plots" \
        -t parca
    """

    stub:
    """
    mkdir plots
    echo -e "${config_uri}\n\n${kb_uri}" > plots/test.txt
    """
}

process createVariants {
    // Parse variants in config JSON to generate variants
    // Outputs directly to publishDir via fsspec and returns variant URIs + hashes

    label "slurm_submit"

    input:
    // Accept URIs and hashes (hashes for cache invalidation)
    tuple val(config_uri), val(config_hash), val(kb_uri), val(kb_hash)

    output:
    // Output variant URIs, hashes, and metadata URI (no file staging for metadata)
    tuple val(config_uri), val(config_hash), path('variant_info.txt'), emit: variantInfo
    env 'metadata_uri', emit: variantMetadataUri

    script:
    def publish_path = "${params.publishDir}/${params.experimentId}/variant_sim_data"
    """
    # Run create_variants.py - it reads kb and writes outputs directly via fsspec
    PYTHONUNBUFFERED=1 python ${params.projectRoot}/runscripts/create_variants.py \\
        --config "${config_uri}" --kb "${kb_uri}" -o "${publish_path}"

    # Read metadata URI from file written by create_variants.py
    export metadata_uri=\$(cat metadata_uri.txt)
    """

    stub:
    def publish_path = "${params.publishDir}/${params.experimentId}/variant_sim_data"
    """
    mkdir -p ${publish_path}
    echo "baseline" > ${publish_path}/0.cPickle
    echo "variant_1" > ${publish_path}/1.cPickle
    echo "variant_2" > ${publish_path}/2.cPickle
    echo '{"null": {"0": "baseline"}}' > ${publish_path}/metadata.json
    echo "${publish_path}/0.cPickle\tmock_hash_0\t0" > variant_info.txt
    echo "${publish_path}/1.cPickle\tmock_hash_1\t1" >> variant_info.txt
    echo "${publish_path}/2.cPickle\tmock_hash_2\t2" >> variant_info.txt
    export metadata_uri=${publish_path}/metadata.json
    """
}

process hqWorker {
    cpus { num_sims * params.sim_cpus }

    memory {
        if ( task.exitStatus in [137, 140] ) {
            1.GB * params.sim_mem * (num_sims + task.attempt - 1)
        } else {
            1.GB * params.sim_mem * num_sims
        }
    }
    time 24.h
    maxRetries 10

    tag "hq_${params.experimentId}_${task.index}"

    label "slurm_submit"
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
        --resource "mem=sum(${task.memory.toMega()})" \\
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
    createVariants(parca_out)
    // Parse variant_info.txt to create channel of (config_uri, config_hash, sim_data_uri, sim_data_hash, variant_name)
    createVariants.out
        .variantInfo
        .map { config_uri, config_hash, variant_file ->
            variant_file.readLines().collect { line ->
                def parts = line.split('\t')
                // variant_info.txt format: sim_data_uri<TAB>sim_data_hash<TAB>variant_name
                tuple(config_uri, config_hash, parts[0], parts[1], parts[2])
            }
        }
        .flatMap { it }
        .set { variantCh }
    createVariants.out
        .variantMetadataUri
        .set { variantMetadataCh }
WORKFLOW
    // Fit as many sims per worker as possible based on available cores
    def simsPerWorker = Math.max(1, params.hq_cores.intdiv(params.sim_cpus))
    if ( params.hyperqueue ) {
        variantCh.combine( seedCh )
            .buffer( size: simsPerWorker, remainder: true )
            .map { it.size() }
            .set { hqChannel }
        hqWorker( hqChannel )
    }
}
