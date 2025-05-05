import COPASI


def _set_initial_concentrations(changes,dm):
    model = dm.getModel()
    assert(isinstance(model, COPASI.CModel))

    references = COPASI.ObjectStdVector()

    for name, value in changes:
        species = model.getMetabolite(name)
        # assert(isinstance(species, COPASI.CMetab))
        if species is None:
            print(f"Species {name} not found in model")
            continue
        species.setInitialConcentration(value)
        references.append(species.getInitialConcentrationReference())

    model.updateInitialValues(references)


def _get_transient_concentration(name, dm):
    model = dm.getModel()
    assert(isinstance(model, COPASI.CModel))
    species = model.getMetabolite(name)
    assert(isinstance(species, COPASI.CMetab))
    if species is None:
        print(f"Species {name} not found in model")
        return None
    return species.getConcentration()
