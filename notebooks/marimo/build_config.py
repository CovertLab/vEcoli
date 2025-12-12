import marimo

__generated_with = "0.14.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import json
    import os

    return json, mo, os


@app.cell
def _(os):
    wd_root = os.getcwd().split("/notebooks")[0]

    return (wd_root,)


@app.cell
def _(json, os, wd_root):
    config_dir = os.path.join(wd_root, "configs")

    with open(os.path.join(config_dir, "default.json"), "r") as f:
        config_default = json.load(f)
    return config_default, config_dir


@app.cell
def _(config_default, generate_config_fields, mo, os, wd_root):
    parca_options_default = config_default["parca_options"]

    parca_label_mapping = {
        "cpus": "Number of CPUs",
        "outdir": "Output directory",
        "operons": "Operons",
        "ribosome_fitting": "Ribosome fitting",
        "rnapoly_fitting": "RNAP fitting",
        "remove_rrna_operons": "Remove rRNA operons",
        "remove_rrff": "Remove rrfF",
        "stable_rrna": "Stable rRNA",
        "gene_deletions": "Gene deletions",
        "new_genes": "New gene options",
        "debug_parca": "Debug",
        "load_intermediate": "Load intermediates",
        "save_intermediates": "Save intermediates",
        "intermediates_directory": "",
        "variable_elongation_transcription": "Variable elongation (transcription)",
        "variable_elongation_translation": "Variable elongation (translation)",
    }

    parca_options_exposed = [
        "cpus",
        "new_genes",
        "operons",
        "ribosome_fitting",
        "rnapoly_fitting",
        "remove_rrna_operons",
        "remove_rrff",
        "stable_rrna",
        "variable_elongation_transcription",
        "variable_elongation_translation",
    ]

    def get_parca_ui(parca_dict):
        parca_ui_dict = {}
        for key in parca_dict.keys():
            key_label = str(parca_label_mapping[key])
            val_default = parca_dict[key]
            if isinstance(val_default, str):
                parca_ui_dict[key] = mo.ui.text(label=key_label, value=val_default)
            elif isinstance(val_default, bool):
                parca_ui_dict[key] = mo.ui.checkbox(label=key_label, value=val_default)
            elif isinstance(val_default, int):
                parca_ui_dict[key] = mo.ui.number(
                    label=key_label, start=1, stop=64, step=1, value=val_default
                )
            new_genes_dir = os.path.join(
                wd_root, "reconstruction", "ecoli", "flat", "new_gene_data"
            )

            new_gene_options = [parca_options_default["new_genes"]]

            new_gene_options = new_gene_options + next(os.walk(new_genes_dir))[1]

            parca_ui_dict["new_genes"] = mo.ui.dropdown(
                options=new_gene_options,
                label=parca_label_mapping["new_genes"],
                value=parca_dict["new_genes"],
            )

        return parca_ui_dict

    parca_form_fields = generate_config_fields(
        parca_options_exposed, get_parca_ui(parca_options_default)
    )

    md_base_str_parca = """
    **ParCa options:**

    """

    fields_list_parca = [
        "{" + str(element) + "}" for element in parca_form_fields.keys()
    ]

    md_body_parca = """

    """.join(fields_list_parca)

    parca_form_md = mo.md(md_base_str_parca + md_body_parca)

    return parca_form_fields, parca_form_md


@app.cell
def _(mo):
    parca_switch = mo.ui.checkbox(label="Run ParCa", value=True)
    parca_switch
    return (parca_switch,)


@app.cell
def _(mo, parca_form_fields, parca_form_md, parca_switch):
    parca_form = None

    if parca_switch.value:
        parca_form = mo.ui.batch(html=parca_form_md, elements=parca_form_fields).form(
            show_clear_button=True, submit_button_label="Apply", bordered=False
        )

    parca_form
    return (parca_form,)


@app.cell
def _(mo):
    variant_switch = mo.ui.checkbox(label="Create variants", value=False)
    variant_switch
    return (variant_switch,)


@app.cell
def _(mo, os, variant_switch, wd_root):
    variant_dir = os.path.join(wd_root, "ecoli", "variants")

    def list_variants(dir):
        variant_files = os.listdir(dir)
        variant_files.remove("__init__.py")
        variant_files.remove("__pycache__")
        variant_files = [file.replace(".py", "") for file in variant_files]
        return variant_files

    variant_modules = list_variants(variant_dir)

    variant_select = None

    if variant_switch.value:
        variant_select = mo.ui.dropdown(options=variant_modules, label="Variant type")

    variant_select
    return (variant_select,)


@app.cell
def _(generate_config_fields, mo):
    def variant_form_conditions():
        condition_options = ["basal", "with_aa", "acetate", "succinate", "no_oxygen"]
        return mo.ui.multiselect(options=condition_options, label="Select condition(s)")

    def load_variant_conditions(var_list):
        condition_dict = {"condition": {"condition": {"value": var_list}}}
        return condition_dict

    def variant_form_new_gene():
        variant_fields = [
            "condition",
            "induction_gen",
            "knockout",
            "knockout_gen",
            "exp_start",
            "exp_stop",
            "exp_num",
            "trl_eff",
            "rel_exp_adj_list",
            "rel_trl_eff_adj_list",
        ]

        variant_fields_label = {
            "condition": "Condition",
            "induction_gen": "Induction generation",
            "knockout": "Knockout",
            "knockout_gen": "Knockout generation",
            "exp_start": "exp_start",
            "exp_stop": "exp_stop",
            "exp_num": "exp_num",
            "trl_eff": "trl_eff",
            "rel_exp_adj_list": "rel_exp_adj_list",
            "rel_trl_eff_adj_list": "rel_trl_eff_adj_list",
        }

        variant_fields_ui = {
            "condition": mo.ui.dropdown(
                label=variant_fields_label["condition"],
                options=["basal", "with_aa", "acetate", "succinate", "no_oxygen"],
            ),
            "induction_gen": mo.ui.number(
                label=variant_fields_label["induction_gen"],
                start=1,
                stop=64,
                step=1,
                value=1,
            ),
            "exp_start": mo.ui.number(
                label=variant_fields_label["exp_start"], start=1, stop=10, step=1
            ),
            "exp_stop": mo.ui.number(
                label=variant_fields_label["exp_stop"], start=1, stop=10, step=1
            ),
            "exp_num": mo.ui.number(
                label=variant_fields_label["exp_num"], start=1, stop=10, step=1
            ),
            "trl_eff": mo.ui.slider(
                label=variant_fields_label["trl_eff"], start=0.0, stop=1.0, step=0.01
            ),
            "rel_exp_adj_list": mo.ui.text(
                label=variant_fields_label["rel_exp_adj_list"]
            ),
            "rel_trl_eff_adj_list": mo.ui.text(
                label=variant_fields_label["rel_trl_eff_adj_list"]
            ),
        }

        variant_fields_ui["knockout"] = mo.ui.checkbox(label="Knockout")
        variant_fields_ui["knockout_gen"] = mo.ui.number(
            label=variant_fields_label["knockout_gen"], start=2, stop=64, step=1
        )

        variant_form_fields = generate_config_fields(variant_fields, variant_fields_ui)

        md_base_str = """
        **Variant Options:**

        """

        fields_list = [
            "{" + str(element) + "}" for element in variant_form_fields.keys()
        ]

        md_body = """

        """.join(fields_list)

        variant_form_md = mo.md(md_base_str + md_body)

        variant_form = mo.ui.batch(
            html=variant_form_md, elements=variant_form_fields
        ).form(show_clear_button=True, submit_button_label="Apply", bordered=False)

        return variant_form

    def load_variant_new_gene(variant_options_input):
        condition_dict = {}
        condition_dict["value"] = []
        condition_dict["value"].append(variant_options_input["condition"])
        induction_gen_dict = {}
        induction_gen_dict["value"] = []
        induction_gen_dict["value"].append(variant_options_input["induction_gen"])
        trl_eff = []
        trl_eff.append(variant_options_input["trl_eff"])
        exp_trl_eff = {
            "nested": {
                "exp": {
                    "logspace": {
                        "start": variant_options_input["exp_start"],
                        "stop": variant_options_input["exp_stop"],
                        "num": variant_options_input["exp_num"],
                    }
                },
                "trl_eff": {"value": trl_eff},
                "op": "prod",
            }
        }
        rel_exp_adj_input = str(variant_options_input["rel_exp_adj_list"])
        rel_exp_adj_list = [float(val) for val in rel_exp_adj_input.split(",")]
        rel_trl_eff_adj_input = str(variant_options_input["rel_trl_eff_adj_list"])
        rel_trl_eff_adj_list = [float(val) for val in rel_trl_eff_adj_input.split(",")]

        rel_adj = {
            "nested": {
                "rel_exp_adj_list": {"value": [rel_exp_adj_list]},
                "rel_trl_eff_adj_list": {"value": [rel_trl_eff_adj_list]},
                "op": "prod",
            }
        }

        variant_options_dict = {}
        variant_options_dict["new_gene_internal_shift_variable_strength"] = {
            "condition": condition_dict,
            "induction_gen": induction_gen_dict,
            "exp_trl_eff": exp_trl_eff,
            "rel_adj": rel_adj,
        }

        if variant_options_input["knockout"]:
            knockout_opt = {}
            knockout_opt["value"] = []
            knockout_opt["value"].append(variant_options_input["knockout_gen"])
            variant_options_dict["new_gene_internal_shift_variable_strength"][
                "knockout_gen"
            ] = knockout_opt

        variant_options_dict["new_gene_internal_shift_variable_strength"]["op"] = "prod"

        return variant_options_dict

    def variant_form_internal_shift():
        variant_fields = [
            "condition",
            "induction_gen",
            "knockout",
            "knockout_gen",
            "exp_start",
            "exp_stop",
            "exp_num",
            "trl_eff",
        ]

        variant_fields_label = {
            "condition": "Condition",
            "induction_gen": "Induction generation",
            "knockout": "Knockout",
            "knockout_gen": "Knockout generation",
            "exp_start": "exp_start",
            "exp_stop": "exp_stop",
            "exp_num": "exp_num",
            "trl_eff": "trl_eff",
            "rel_exp_adj_list": "rel_exp_adj_list",
            "rel_trl_eff_adj_list": "rel_trl_eff_adj_list",
        }

        variant_fields_ui = {
            "condition": mo.ui.dropdown(
                label=variant_fields_label["condition"],
                options=["basal", "with_aa", "acetate", "succinate", "no_oxygen"],
            ),
            "induction_gen": mo.ui.number(
                label=variant_fields_label["induction_gen"],
                start=1,
                stop=64,
                step=1,
                value=1,
            ),
            "exp_start": mo.ui.number(
                label=variant_fields_label["exp_start"], start=1, stop=10, step=1
            ),
            "exp_stop": mo.ui.number(
                label=variant_fields_label["exp_stop"], start=1, stop=10, step=1
            ),
            "exp_num": mo.ui.number(
                label=variant_fields_label["exp_num"], start=1, stop=10, step=1
            ),
            "trl_eff": mo.ui.slider(
                label=variant_fields_label["trl_eff"], start=0.0, stop=1.0, step=0.01
            ),
            "rel_exp_adj_list": mo.ui.text(
                label=variant_fields_label["rel_exp_adj_list"]
            ),
            "rel_trl_eff_adj_list": mo.ui.text(
                label=variant_fields_label["rel_trl_eff_adj_list"]
            ),
        }

        variant_fields_ui["knockout"] = mo.ui.checkbox(label="Knockout")
        variant_fields_ui["knockout_gen"] = mo.ui.number(
            label=variant_fields_label["knockout_gen"], start=2, stop=64, step=1
        )

        variant_form_fields = generate_config_fields(variant_fields, variant_fields_ui)

        md_base_str = """
        **Variant Options:**

        """

        fields_list = [
            "{" + str(element) + "}" for element in variant_form_fields.keys()
        ]

        md_body = """

        """.join(fields_list)

        variant_form_md = mo.md(md_base_str + md_body)

        variant_form = mo.ui.batch(
            html=variant_form_md, elements=variant_form_fields
        ).form(show_clear_button=True, submit_button_label="Apply", bordered=False)

        return variant_form

    def load_variant_internal_shift(variant_options_input):
        condition_dict = {}
        condition_dict["value"] = []
        condition_dict["value"].append(variant_options_input["condition"])
        induction_gen_dict = {}
        induction_gen_dict["value"] = []
        induction_gen_dict["value"].append(variant_options_input["induction_gen"])
        trl_eff = []
        trl_eff.append(variant_options_input["trl_eff"])
        exp_trl_eff = {
            "nested": {
                "exp": {
                    "logspace": {
                        "start": variant_options_input["exp_start"],
                        "stop": variant_options_input["exp_stop"],
                        "num": variant_options_input["exp_num"],
                    }
                },
                "trl_eff": {"value": trl_eff},
                "op": "prod",
            }
        }

        variant_options_dict = {}
        variant_options_dict["new_gene_internal_shift"] = {
            "condition": condition_dict,
            "induction_gen": induction_gen_dict,
            "exp_trl_eff": exp_trl_eff,
        }

        if variant_options_input["knockout"]:
            knockout_opt = {}
            knockout_opt["value"] = []
            knockout_opt["value"].append(variant_options_input["knockout_gen"])
            variant_options_dict["new_gene_internal_shift"]["knockout_gen"] = (
                knockout_opt
            )

        variant_options_dict["new_gene_internal_shift"]["op"] = "prod"

        return variant_options_dict

    return (
        load_variant_conditions,
        load_variant_internal_shift,
        load_variant_new_gene,
        variant_form_conditions,
        variant_form_internal_shift,
        variant_form_new_gene,
    )


@app.cell
def _(
    load_variant_conditions,
    load_variant_internal_shift,
    load_variant_new_gene,
    variant_form_conditions,
    variant_form_internal_shift,
    variant_form_new_gene,
):
    variant_forms = {
        "condition": variant_form_conditions,
        "new_gene_internal_shift": variant_form_internal_shift,
        "new_gene_internal_shift_variable_strength": variant_form_new_gene,
    }

    variant_loader = {
        "condition": load_variant_conditions,
        "new_gene_internal_shift": load_variant_internal_shift,
        "new_gene_internal_shift_variable_strength": load_variant_new_gene,
    }
    return variant_forms, variant_loader


@app.cell
def _(variant_forms, variant_select, variant_switch):
    variant_params_input = None
    if variant_switch.value:
        variant_params_input = variant_forms[variant_select.value]()
    variant_params_input
    return (variant_params_input,)


@app.cell
def _(variant_loader, variant_params_input, variant_select, variant_switch):
    variant_options = {}

    if variant_switch.value:
        if not isinstance(variant_params_input, type(None)):
            if variant_params_input.value:
                variant_options["variants"] = variant_loader[variant_select.value](
                    variant_params_input.value
                )

    return (variant_options,)


@app.cell
def _(config_default, mo, parca_switch, variant_switch):
    field_label_mapping = {
        "experiment_id": "Experiment ID",
        "suffix_time": "Suffix (time)",
        "emitter": "Emitter",
        "generations": "Number of generations",
        "n_init_sims": "Number of initial seeds",
        "max_duration": "Maximum duration (seconds)",
        "single_daughters": "Single daughters",
        "fail_at_max_duration": "Fail at max duration",
        "skip_baseline": "Skip baseline",
    }

    field_ui_mapping = {
        "experiment_id": mo.ui.text(label=f"{field_label_mapping['experiment_id']}"),
        "suffix_time": mo.ui.checkbox(
            label=f"{field_label_mapping['suffix_time']}",
            value=config_default["suffix_time"],
        ),
        "emitter": mo.ui.dropdown(
            label=f"{field_label_mapping['emitter']}",
            options=["parquet", "timeseries"],
            value="parquet",
        ),
        "generations": mo.ui.number(
            label=f"{field_label_mapping['generations']}", start=1, stop=40, step=1
        ),
        "n_init_sims": mo.ui.number(
            label=f"{field_label_mapping['n_init_sims']}", start=1, stop=100, step=1
        ),
        "max_duration": mo.ui.slider(
            label=f"{field_label_mapping['max_duration']}",
            start=0.0,
            stop=10800.0,
            step=1.0,
            value=float(config_default["max_duration"]),
        ),
        "single_daughters": mo.ui.checkbox(
            label=f"{field_label_mapping['single_daughters']}",
            value=config_default["single_daughters"],
        ),
        "fail_at_max_duration": mo.ui.checkbox(
            label=f"{field_label_mapping['fail_at_max_duration']}"
        ),
        "skip_baseline": mo.ui.checkbox(label=field_label_mapping["skip_baseline"]),
    }

    fields_exposed = [
        "experiment_id",
        "sim_data_path",
        "emitter",
        "generations",
        "n_init_sims",
        "max_duration",
        "suffix_time",
        "single_daughters",
        "fail_at_max_duration",
        "skip_baseline",
    ]
    if parca_switch.value:
        fields_exposed.remove("sim_data_path")
    if not variant_switch.value:
        fields_exposed.remove("skip_baseline")
    return field_ui_mapping, fields_exposed


@app.cell
def _(mo):
    def generate_config_fields(config_keys, field_ui_mapping):
        config_form_fields = {}

        for key in config_keys:
            ui_element = field_ui_mapping.get(key)
            if isinstance(ui_element, type(None)):
                ui_element = mo.ui.text(label=f"{key}: ")
            config_form_fields[key] = ui_element

        return config_form_fields

    return (generate_config_fields,)


@app.cell
def _(field_ui_mapping, fields_exposed, generate_config_fields, mo):
    config_form_fields = generate_config_fields(fields_exposed, field_ui_mapping)

    md_base_str = """
    **Simulation Configuration:**

    """

    fields_list = ["{" + str(element) + "}" for element in config_form_fields.keys()]

    md_body = """

    """.join(fields_list)

    config_form_md = mo.md(md_base_str + md_body)
    return config_form_fields, config_form_md


@app.cell
def _(config_form_fields, config_form_md, mo):
    config_form = mo.ui.batch(html=config_form_md, elements=config_form_fields).form(
        show_clear_button=True, submit_button_label="Apply", bordered=False
    )

    config_form
    return (config_form,)


@app.cell
def _(mo, os, wd_root):
    # analysis options
    analysis_switch = mo.ui.checkbox(label="Run analysis")

    def get_analysis_modules(analysis_type, root=wd_root):
        dir_analysis = os.path.join(root, "ecoli", "analysis")
        modules = os.listdir(os.path.join(dir_analysis, analysis_type))
        if "__init__.py" in modules:
            modules.remove("__init__.py")
        if "__pycache__" in modules:
            modules.remove("__pycache__")
        if len(modules) > 0:
            modules = [name.split(".py")[0] for name in modules]
        return modules

    def get_analysis_menu(analysis_type):
        menu = mo.ui.multiselect(
            options=get_analysis_modules(analysis_type),
            label=f"Analysis ({analysis_type})",
        )
        return menu

    return analysis_switch, get_analysis_menu


@app.cell
def _(analysis_switch):
    analysis_switch
    return


@app.cell
def _(analysis_switch, mo):
    analysis_type_select = None
    if analysis_switch.value:
        analysis_type_select = mo.ui.multiselect(
            options=[
                "single",
                "multidaughter",
                "multigeneration",
                "multiseed",
                "multivariant",
            ],
            label="Analysis type",
        )
    analysis_type_select
    return (analysis_type_select,)


@app.cell
def _(analysis_type_select, get_analysis_menu):
    analysis_menu_single = None
    analysis_menu_multidaughter = None
    analysis_menu_multigeneration = None
    analysis_menu_multiseed = None
    analysis_menu_multivariant = None

    if not isinstance(analysis_type_select, type(None)):
        if "single" in analysis_type_select.value:
            analysis_menu_single = get_analysis_menu("single")
        if "multidaughter" in analysis_type_select.value:
            analysis_menu_multidaughter = get_analysis_menu("multidaughter")
        if "multigeneration" in analysis_type_select.value:
            analysis_menu_multigeneration = get_analysis_menu("multigeneration")
        if "multiseed" in analysis_type_select.value:
            analysis_menu_multiseed = get_analysis_menu("multiseed")
        if "multivariant" in analysis_type_select.value:
            analysis_menu_multivariant = get_analysis_menu("multivariant")
    return (
        analysis_menu_multidaughter,
        analysis_menu_multigeneration,
        analysis_menu_multiseed,
        analysis_menu_multivariant,
        analysis_menu_single,
    )


@app.cell
def _(analysis_menu_single):
    analysis_menu_single
    return


@app.cell
def _(analysis_menu_multidaughter):
    analysis_menu_multidaughter
    return


@app.cell
def _(analysis_menu_multigeneration):
    analysis_menu_multigeneration
    return


@app.cell
def _(analysis_menu_multiseed):
    analysis_menu_multiseed
    return


@app.cell
def _(analysis_menu_multivariant):
    analysis_menu_multivariant
    return


@app.cell
def _(
    analysis_menu_multidaughter,
    analysis_menu_multigeneration,
    analysis_menu_multiseed,
    analysis_menu_multivariant,
    analysis_menu_single,
):
    analysis_options = {}
    analysis_options_single = {}
    analysis_options_multidaughter = {}
    analysis_options_multigeneration = {}
    analysis_options_multiseed = {}
    analysis_options_multivariant = {}

    if not isinstance(analysis_menu_single, type(None)):
        if len(analysis_menu_single.value) > 0:
            analysis_options_single = {
                module: {} for module in analysis_menu_single.value
            }
            analysis_options["single"] = analysis_options_single

    if not isinstance(analysis_menu_multidaughter, type(None)):
        if len(analysis_menu_multidaughter.value) > 0:
            analysis_options_multidaughter = {
                module: {} for module in analysis_menu_multidaughter.value
            }
            analysis_options["multidaughter"] = analysis_options_multidaughter

    if not isinstance(analysis_menu_multigeneration, type(None)):
        if len(analysis_menu_multigeneration.value) > 0:
            analysis_options_multigeneration = {
                module: {} for module in analysis_menu_multigeneration.value
            }
            analysis_options["multigeneration"] = analysis_options_multigeneration

    if not isinstance(analysis_menu_multiseed, type(None)):
        if len(analysis_menu_multiseed.value) > 0:
            analysis_options_multiseed = {
                module: {} for module in analysis_menu_multiseed.value
            }
            analysis_options["multiseed"] = analysis_options_multiseed

    if not isinstance(analysis_menu_multivariant, type(None)):
        if len(analysis_menu_multivariant.value) > 0:
            analysis_options_multivariant = {
                module: {} for module in analysis_menu_multivariant.value
            }
            analysis_options["multivariant"] = analysis_options_multivariant

    analysis_dict = {"analysis_options": analysis_options}
    return analysis_dict, analysis_options


@app.cell
def _(parca_form):
    emitter_options = {}

    emitter_options["emitter_arg"] = {}

    emitter_options["emitter_arg"]["out_dir"] = "out"

    parca_options = {}

    parca_options["parca_options"] = parca_form.value
    return emitter_options, parca_options


@app.cell
def _(
    analysis_dict,
    analysis_options,
    analysis_switch,
    config_form,
    emitter_options,
    parca_options,
    parca_switch,
    variant_options,
    variant_switch,
):
    config_final = config_form.value | emitter_options
    if parca_switch.value:
        config_final["sim_data_path"] = None
        config_final = config_final | parca_options

    if variant_switch.value:
        if len(variant_options) > 0:
            config_final = config_final | variant_options

    if analysis_switch.value:
        if len(analysis_options) > 0:
            config_final = config_final | analysis_dict
    return (config_final,)


@app.cell
def _(json, os):
    def save_config(filename, config_out_dir, final_config_dict):
        if not filename.endswith(".json"):
            filename = filename + ".json"

        with open(os.path.join(config_out_dir, filename), "w") as f:
            json.dump(final_config_dict, f, indent=4)

    return (save_config,)


@app.cell
def _(mo):
    config_save_name = mo.ui.text(label="Save config as: ", value="test_config")
    config_save_name
    return (config_save_name,)


@app.cell
def _(mo):
    save_button = mo.ui.run_button(label="save config")
    save_button
    return (save_button,)


@app.cell
def _(mo):
    get_button_val, set_button_val = mo.state(0)
    return


@app.cell
def _(
    config_dir,
    config_final,
    config_save_name,
    mo,
    save_button,
    save_config,
):
    mo.stop(not save_button.value)
    if len(config_save_name.value) > 0:
        save_config(config_save_name.value, config_dir, config_final)

    return


if __name__ == "__main__":
    app.run()
