process runParca {
    // Run ParCa using parca_options from config JSON.
    // Outputs directly to publishDir via fsspec and returns URIs + hash for
    // downstream processes.
    //
    // parca_id distinguishes parallel ParCa runs (e.g. different RNA-seq
    // datasets in a multi-parca workflow). For single-parca workflows it is 0.
    // offset is the index shift applied by the downstream createVariants
    // process so variants from different parcas occupy non-overlapping
    // index ranges; 0 for single-parca.

    label "parca"

    input:
    tuple val(parca_id), val(config_uri), path(config), val(offset)

    output:
    // (parca_id, config_uri, config_hash, kb_uri, kb_hash, offset)
    // config_hash is the sha256 of the *per-parca* config so cache invalidation
    // is per-parca-correct even though config_uri may be the global config.
    tuple val(parca_id), val(config_uri), env('config_hash'), val("${params.publishDir}/${params.experimentId}/parca_${parca_id}/kb"), env('kb_hash'), val(offset), emit: parca_out

    script:
    def publish_path = "${params.publishDir}/${params.experimentId}/parca_${parca_id}"
    """
    # Belt-and-suspenders: source /vEcoli/.env so ECOLI_SOURCES resolves
    # correctly even when the container's ENTRYPOINT is bypassed
    # (e.g. K8s `command:` override, some Batch executor configs).
    [[ -f /vEcoli/.env ]] && set -a && source /vEcoli/.env && set +a || true

    # Compute config hash for cache invalidation (per-parca config content)
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
    def publish_path = "${params.publishDir}/${params.experimentId}/parca_${parca_id}/kb"
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
    [[ -f /vEcoli/.env ]] && set -a && source /vEcoli/.env && set +a || true
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
    // Parse variants in config JSON to generate variants.
    // Outputs directly to publishDir via fsspec and returns variant URIs + hashes.
    //
    // parca_id + offset enable multi-parca: each parca's variants land at
    // distinct indices, and metadata is namespaced by parca_id so that
    // mergeVariantMetadata can deep-merge them.

    label "slurm_submit"

    input:
    tuple val(parca_id), val(config_uri), val(config_hash), val(kb_uri), val(kb_hash), val(offset)

    output:
    // (config_uri, config_hash, variant_info_path)  — variant_info.txt rows
    // are offset-shifted, so flattening across parcas yields globally unique
    // variant indices.
    tuple val(config_uri), val(config_hash), path('variant_info.txt'), emit: variantInfo
    // Per-parca metadata file URI; mergeVariantMetadata collects all of them
    // and deep-merges via fsspec. URI-based (not staged) because the file
    // lives on cloud storage written directly by create_variants.py.
    env 'metadata_uri', emit: variantMetadataUri

    script:
    def publish_path = "${params.publishDir}/${params.experimentId}/variant_sim_data"
    """
    # Belt-and-suspenders: source /vEcoli/.env so ECOLI_SOURCES resolves
    # correctly even when the container's ENTRYPOINT is bypassed.
    [[ -f /vEcoli/.env ]] && set -a && source /vEcoli/.env && set +a || true

    PYTHONUNBUFFERED=1 python ${params.projectRoot}/runscripts/create_variants.py \\
        --config "${config_uri}" --kb "${kb_uri}" --offset ${offset} \\
        --parca-id ${parca_id} \\
        -o "${publish_path}"

    # create_variants.py writes metadata_<parca_id>.json directly to
    # publish_path via fsspec; it leaves metadata_uri.txt locally for
    # Nextflow to slurp into the env emit.
    export metadata_uri=\$(cat metadata_uri.txt)
    """

    stub:
    def publish_path = "${params.publishDir}/${params.experimentId}/variant_sim_data"
    """
    mkdir -p ${publish_path}
    echo "baseline" > ${publish_path}/${offset}.cPickle
    echo "variant_${parca_id}_1" > ${publish_path}/\$((${offset} + 1)).cPickle
    echo "variant_${parca_id}_2" > ${publish_path}/\$((${offset} + 2)).cPickle
    echo '{"null": {"${offset}": "baseline"}}' > ${publish_path}/metadata_${parca_id}.json
    echo "${publish_path}/${offset}.cPickle\tmock_hash_${parca_id}_0\t${offset}" > variant_info.txt
    echo "${publish_path}/\$((${offset} + 1)).cPickle\tmock_hash_${parca_id}_1\t\$((${offset} + 1))" >> variant_info.txt
    echo "${publish_path}/\$((${offset} + 2)).cPickle\tmock_hash_${parca_id}_2\t\$((${offset} + 2))" >> variant_info.txt
    export metadata_uri=${publish_path}/metadata_${parca_id}.json
    """
}


process mergeVariantMetadata {
    // Combine per-parca metadata_<parca_id>.json files into a unified
    // metadata.json that downstream analyses expect. For single-parca
    // workflows this is a near-no-op (one URI in, same content re-written).
    //
    // Deep-merges by variant_name so the same variant key across parcas
    // gets its variant_idx -> params dicts combined (rather than overwritten).
    // Input is a LIST of URIs (collected from per-parca createVariants runs)
    // and each file is fetched via fsspec — no local staging.

    label "slurm_submit"

    input:
    val metadata_uris

    output:
    env 'metadata_uri', emit: variantMetadataUri

    script:
    def publish_path = "${params.publishDir}/${params.experimentId}/variant_sim_data"
    def uri_list = metadata_uris instanceof List ? metadata_uris : [metadata_uris]
    def uri_args = uri_list.collect { "\"${it}\"" }.join(' ')
    """
    python <<PY
import json, fsspec, sys
uris = [${uri_list.collect { "'${it}'" }.join(', ')}]
merged = {}
for uri in uris:
    with fsspec.open(uri, 'r') as fh:
        for vname, vmeta in json.load(fh).items():
            merged.setdefault(vname, {}).update(vmeta)
with fsspec.open('${publish_path}/metadata.json', 'w') as fh:
    json.dump(merged, fh)
PY
    export metadata_uri='${publish_path}/metadata.json'
    """

    stub:
    def publish_path = "${params.publishDir}/${params.experimentId}/variant_sim_data"
    """
    mkdir -p ${publish_path}
    echo '{"null": {"0": "baseline"}}' > ${publish_path}/metadata.json
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
    // parca_out_full carries (parca_id, config_uri, config_hash, kb_uri, kb_hash, offset)
    // — one element per parca run. createVariants fans out automatically.
    createVariants(parca_out_full)
    // variantInfo emits (config_uri, config_hash, variant_info.txt) per parca;
    // flattening across parcas produces globally unique (offset-shifted) variant rows.
    createVariants.out
        .variantInfo
        .map { config_uri, config_hash, variant_file ->
            variant_file.readLines().collect { line ->
                def parts = line.split('\t')
                // variant_info.txt format: sim_data_uri<TAB>sim_data_hash<TAB>variant_idx
                tuple(config_uri, config_hash, parts[0], parts[1], parts[2])
            }
        }
        .flatMap { it }
        .set { variantCh }
    // Collect per-parca metadata URIs and merge into one metadata.json.
    // For single-parca runs this is a 1-in-1-out pass-through.
    mergeVariantMetadata(createVariants.out.variantMetadataUri.collect())
    mergeVariantMetadata.out.variantMetadataUri.set { variantMetadataCh }
    // Single tuple for cross-parca analyses (analysisMultiVariant, analysisParca,
    // ...) — use parca_0's, since validationData doesn't depend on rnaseq_options.
    // .first() converts to a value channel so downstream analyses can re-read it.
    parca_out_full
        .filter { it[0] == 0 }
        .map { parca_id, c_uri, c_hash, k_uri, k_hash, off -> tuple(c_uri, c_hash, k_uri, k_hash) }
        .first()
        .set { parca_out }
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
