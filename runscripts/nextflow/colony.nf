process colony {
    input:
    path config
    path sim_data
    path initial_state

    output:
    env STATUS

    script:
    """
    uv --project ${params.projectRoot} run --env-file ${params.projectRoot}/.env \\
        ${params.projectRoot}/ecoli/experiments/ecoli_engine_process.py \\
        --config $config --sim_data_path $sim_data --initial_state_file $initial_state
    STATUS=$?
    """

    stub:
    """
    STATUS=0
    """
}
