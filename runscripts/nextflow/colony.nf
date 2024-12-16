process colony {
    input:
    path config
    path sim_data
    path initial_state

    output:
    env STATUS

    script:
    """
    python /vEcoli/ecoli/experiments/ecoli_engine_process.py --config $config --sim_data_path $sim_data --initial_state_file $initial_state
    STATUS=$?
    """

    stub:
    """
    STATUS=0
    """
}
