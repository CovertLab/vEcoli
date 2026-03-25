process runParca {
    // Run ParCa using parca_options from config JSON.
    // parca_id distinguishes runs when multiple ParCa instances are used
    // (e.g. different RNAseq datasets). For single-parca workflows parca_id=0.
    publishDir { "${params.publishDir}/${params.experimentId}/parca_${parca_id}" }, mode: "copy"

    label "parca"

    input:
    tuple val(parca_id), path(config)

    output:
    tuple val(parca_id), path('kb')

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
    publishDir { "${params.publishDir}/${params.experimentId}/parca/analysis" }, mode: "copy"

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
    // Parse variants in config JSON to generate variants.
    // offset shifts all variant indices so that variants from different ParCa
    // runs occupy non-overlapping index ranges within the same experiment.
    publishDir "${params.publishDir}/${params.experimentId}/variant_sim_data", mode: "copy"

    label "slurm_submit"

    input:
    path config
    tuple val(parca_id), path(kb), val(offset)

    output:
    path '*.cPickle', emit: variantSimData
    path "metadata_${parca_id}.json", emit: variantMetadata

    script:
    """
    python ${params.projectRoot}/runscripts/create_variants.py \
        --config "$config" --kb "$kb" --offset $offset -o "\$(pwd)"
    mv metadata.json metadata_${parca_id}.json
    """

    stub:
    """
    cp $kb/simData.cPickle ${offset}.cPickle
    echo "Mock metadata" > metadata_${parca_id}.json
    """
}

process mergeVariantMetadata {
    // Merge metadata JSON files from multiple createVariants runs into one.
    // For single-parca workflows this is a pass-through (one file in, one out).
    publishDir "${params.publishDir}/${params.experimentId}/variant_sim_data", mode: "copy"

    input:
    path 'metadata_*.json'

    output:
    path 'metadata.json'

    script:
    """
    python -c "
import json, glob
merged = {}
for f in sorted(glob.glob('metadata_*.json')):
    with open(f) as fh:
        merged.update(json.load(fh))
with open('metadata.json', 'w') as fh:
    json.dump(merged, fh)
"
    """

    stub:
    """
    python -c "
import json, glob
merged = {}
for f in sorted(glob.glob('metadata_*.json')):
    with open(f) as fh:
        merged.update(json.load(fh))
with open('metadata.json', 'w') as fh:
    json.dump(merged, fh)
"
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
