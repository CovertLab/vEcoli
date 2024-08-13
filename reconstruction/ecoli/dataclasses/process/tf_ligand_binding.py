"""
SimulationData for TF-ligand binding (considers different TFs
than equilibrium, with different (a little less mechanistic)
modeling to enable more robust TF simulations).
"""

from typing import Union

import numpy as np
from wholecell.utils import data, units

class TFLigandBindingError(Exception):
    pass


class MoleculeNotFoundError(TFLigandBindingError):
    pass


class TFLigandBinding(object):
    """
    SimulationData for TF-ligand binding
    """

    def __init__(self, raw_data, sim_data):
        # Build the abstractions needed for complexation
        molecules = []

        reaction_ids = []

        stoichMatrixI = []
        stoichMatrixJ = []
        stoichMatrixV = []

        stoichMatrixMass = []

        ligand_ids = []
        bound_tf_ids = []
        unbound_tf_ids = []

        # Make sure reactions are not duplicated with complexationReactions or
        # equilibriumReactions
        tf_ligand_binding_reaction_ids = {x["id"] for x in raw_data.tf_ligand_binding_reactions}
        equilibrium_reaction_ids = {x["id"] for x in raw_data.equilibrium_reactions}
        complexation_reaction_ids = {x["id"] for x in raw_data.complexation_reactions}

        # Get reaction binding params
        self.reaction_kds = [x['Kd'] for x in raw_data.tf_ligand_binding_reaction_params]
        self.reaction_hill = [x['Hill coefficient'] for x in raw_data.tf_ligand_binding_reaction_params]

        if tf_ligand_binding_reaction_ids.intersection(complexation_reaction_ids) != set():
            raise Exception(
                "The following reaction ids are specified in tfLigandBindingReactions and complexationReactions: %s"
                % (tf_ligand_binding_reaction_ids.intersection(complexation_reaction_ids))
            )

        if tf_ligand_binding_reaction_ids.intersection(equilibrium_reaction_ids) != set():
            raise Exception(
                "The following reaction ids are specified in tfLigandBindingReactions and equilibriumReactions: %s"
                % (tf_ligand_binding_reaction_ids.intersection(equilibrium_reaction_ids))
            )

        # Get IDs of all metabolites
        metabolite_ids = {met["id"] for met in raw_data.metabolites}

        # IDs of 2CS ligands that should be tagged to the periplasm
        two_component_system_ligands = [
            system["molecules"]["LIGAND"] for system in raw_data.two_component_systems
        ]

        # Remove complexes that are currently not simulated
        FORBIDDEN_MOLECULES = {
            "modified-charged-selC-tRNA",  # molecule does not exist
        }

        # Remove reactions that we know won't occur (e.g., don't do
        # computations on metabolites that have zero counts)
        # TODO (ggsun): check if this list is accurate
        MOLECULES_THAT_WILL_EXIST_IN_SIMULATION = (
                [m["Metabolite"] for m in raw_data.metabolite_concentrations]
                + ["LEU", "S-ADENOSYLMETHIONINE", "ARABINOSE", "4FE-4S"]
                + two_component_system_ligands
        )

        reaction_index = 0

        def should_skip_reaction(reaction):
            for mol_id in reaction["stoichiometry"].keys():
                if mol_id in FORBIDDEN_MOLECULES or (
                        mol_id in metabolite_ids
                        and mol_id not in MOLECULES_THAT_WILL_EXIST_IN_SIMULATION
                ):
                    return True
            return False

        # Build stoichiometry matrix
        for reaction in raw_data.tf_ligand_binding_reactions:
            if should_skip_reaction(reaction):
                continue

            reaction_ids.append(reaction["id"])

            for mol_id, coeff in reaction["stoichiometry"].items():

                # Assume coefficients given as null are -1
                if coeff is None:
                    coeff = -1

                # All stoichiometric coefficients must be integers
                assert coeff % 1 == 0

                if mol_id in metabolite_ids:
                    if mol_id in two_component_system_ligands:
                        mol_id_with_compartment = "{}[{}]".format(
                            mol_id,
                            "p",  # Assume 2CS ligands are in periplasm
                        )
                    else:
                        mol_id_with_compartment = "{}[{}]".format(
                            mol_id,
                            "c",  # Assume all other metabolites are in cytosol
                        )
                    ligand_ids.append(mol_id_with_compartment)
                else:
                    mol_id_with_compartment = "{}[{}]".format(
                        mol_id, sim_data.getter.get_compartment(mol_id)[0]
                    )
                    # If the coefficient is positive, the molecule is the bound TF.
                    # Otherwise, the molecule is the unbound TF.
                    # TODO: any checks that this is true?
                    if coeff > 0:
                        bound_tf_ids.append(mol_id_with_compartment)
                    else:
                        unbound_tf_ids.append(mol_id_with_compartment)

                if mol_id_with_compartment not in molecules:
                    molecules.append(mol_id_with_compartment)
                    molecule_index = len(molecules) - 1
                else:
                    molecule_index = molecules.index(mol_id_with_compartment)

                # Store indices for the row and column, and molecule
                # coefficient for building the stoichiometry matrix
                stoichMatrixI.append(molecule_index)
                stoichMatrixJ.append(reaction_index)
                stoichMatrixV.append(coeff)

                # Find molecular mass
                molecularMass = sim_data.getter.get_mass(
                    mol_id_with_compartment
                ).asNumber(units.g / units.mol)
                stoichMatrixMass.append(molecularMass)

            reaction_index += 1

        # TODO(jerry): Move the rest to a subroutine for __init__ and __setstate__?
        self._stoichMatrixI = np.array(stoichMatrixI)
        self._stoichMatrixJ = np.array(stoichMatrixJ)
        self._stoichMatrixV = np.array(stoichMatrixV)

        self.molecule_names = molecules
        self.bound_tf_idxs = np.array([self.molecule_names.index(x) for x in bound_tf_ids])
        self.unbound_tf_idxs = np.array([self.molecule_names.index(x) for x in unbound_tf_ids])
        self.ligand_idxs = np.array([self.molecule_names.index(x) for x in ligand_ids])

        # TODO (Albert): should probably just save either the ids or the idxs as attributes, not both
        self.reaction_ids = reaction_ids
        self.ligand_ids = ligand_ids
        self.bound_tf_ids = bound_tf_ids
        self.unbound_tf_ids = unbound_tf_ids

        # Mass balance matrix
        self._stoichMatrixMass = np.array(stoichMatrixMass)
        self.balance_matrix = self.stoich_matrix() * self.mass_matrix()

        # Find the mass balance of each equation in the balanceMatrix
        massBalanceArray = self.mass_balance()

        # The stoichometric matrix should balance out to numerical zero.
        assert np.max(np.absolute(massBalanceArray)) < 1e-9

        # Build matrices
        self._stoichMatrix = self.stoich_matrix()
        self._make_matrices()

    def __getstate__(self):
        """Return the state to pickle, omitting derived attributes that
        __setstate__() will recompute, esp. those like the rates for ODEs
        that don't pickle.
        """
        return data.dissoc_strict(
            self.__dict__,
            (
                "_stoichMatrix",
                "Rp",
                "Pp",
            ),
        )

    def __setstate__(self, state):
        """Restore instance attributes, recomputing some of them."""
        self.__dict__.update(state)
        self._stoichMatrix = self.stoich_matrix()
        self._make_matrices()

    def ligand_bound_fraction(self, ligand_conc):
        # TODO: make the cutoff in a systematic way rather than just 0.05
        active_fracs = []
        for i, conc in enumerate(ligand_conc):
            reaction_id = self.reaction_ids[i]
            if reaction_id == 'CPLX-123_RXN':
                # purR-hypoxanthine reaction
                active_frac = 1 / (1 + (self.reaction_kds[i]/conc)**self.reaction_hill[i])
                # A Hill curve that starts at 0 where active_frac would have been 0.05/1.05,
                # and rises up to 1.
                active_frac = max(0, active_frac*1.05 - 0.05)
            else:
                active_frac = 1 / (1 + (self.reaction_kds[i]/conc)**self.reaction_hill[i])
            active_fracs.append(active_frac)

        return active_fracs

    def stoich_matrix(self):
        """
        Builds stoichiometry matrix
        Rows: molecules
        Columns: reactions
        Values: reaction stoichiometry
        """
        shape = (self._stoichMatrixI.max() + 1, self._stoichMatrixJ.max() + 1)
        out = np.zeros(shape, np.float64)
        out[self._stoichMatrixI, self._stoichMatrixJ] = self._stoichMatrixV
        return out

    def mass_matrix(self):
        """
        Builds stoichiometry mass matrix
        Rows: molecules
        Columns: reactions
        Values: molecular mass
        """
        shape = (self._stoichMatrixI.max() + 1, self._stoichMatrixJ.max() + 1)
        out = np.zeros(shape, np.float64)
        out[self._stoichMatrixI, self._stoichMatrixJ] = self._stoichMatrixMass
        return out

    def mass_balance(self):
        """
        Sum along the columns of the massBalance matrix to check for reaction
        mass balance
        """
        return np.sum(self.balance_matrix, axis=0)

    def stoich_matrix_monomers(self):
        """
        Builds a stoichiometric matrix where each column is a reaction that
        forms a complex directly from its constituent monomers. This will be
        different from stoich_matrix if some reactions in tf_ligand_binding
        involve the binding of a ligand to a complex which itself is also
        formed through ligand binding in tf_ligand_binding.
        """
        stoichMatrixMonomersI = []
        stoichMatrixMonomersJ = []
        stoichMatrixMonomersV = []

        for colIdx, id_complex in enumerate(self.bound_tf_ids):
            D = self.get_monomers(id_complex)

            rowIdx = self.molecule_names.index(id_complex)
            stoichMatrixMonomersI.append(rowIdx)
            stoichMatrixMonomersJ.append(colIdx)
            stoichMatrixMonomersV.append(1.0)

            for subunitId, subunitStoich in zip(D["subunitIds"], D["subunitStoich"]):
                rowIdx = self.molecule_names.index(subunitId)
                stoichMatrixMonomersI.append(rowIdx)
                stoichMatrixMonomersJ.append(colIdx)
                stoichMatrixMonomersV.append(-1.0 * subunitStoich)

        stoichMatrixMonomersI = np.array(stoichMatrixMonomersI)
        stoichMatrixMonomersJ = np.array(stoichMatrixMonomersJ)
        stoichMatrixMonomersV = np.array(stoichMatrixMonomersV)
        shape = (stoichMatrixMonomersI.max() + 1, stoichMatrixMonomersJ.max() + 1)

        out = np.zeros(shape, np.float64)
        out[stoichMatrixMonomersI, stoichMatrixMonomersJ] = stoichMatrixMonomersV
        return out

    def get_monomers(self, cplxId):
        """
        Returns subunits for a complex (or any ID passed). If the ID passed is
        already a monomer returns the monomer ID again with a stoichiometric
        coefficient of one.
        """
        info = self._moleculeRecursiveSearch(
            cplxId, self._stoichMatrix, self.molecule_names
        )
        return {
            "subunitIds": np.array(list(info.keys())),
            "subunitStoich": np.array(list(info.values())),
        }

    def _findRow(self, product, speciesList):
        try:
            row = speciesList.index(product)
        except ValueError as e:
            raise MoleculeNotFoundError(
                "Could not find %s in the list of molecules." % (product,), e
            )
        return row

    def _findColumn(self, stoichMatrixRow):
        for i in range(0, len(stoichMatrixRow)):
            if int(stoichMatrixRow[i]) == 1:
                return i
        return -1  # Flag for monomer

    def _moleculeRecursiveSearch(self, product, stoichMatrix, speciesList):
        row = self._findRow(product, speciesList)
        col = self._findColumn(stoichMatrix[row, :])
        if col == -1:
            return {product: 1.0}

        total = {}
        for i in range(0, len(speciesList)):
            if i == row:
                continue
            val = stoichMatrix[i][col]
            species = speciesList[i]

            if val != 0:
                x = self._moleculeRecursiveSearch(species, stoichMatrix, speciesList)
                for j in x:
                    if j in total:
                        total[j] += x[j] * (np.absolute(val))
                    else:
                        total[j] = x[j] * (np.absolute(val))
        return total

    def _make_matrices(self):
        EPS = 1e-9

        S = self.stoich_matrix()
        Rp = -1.0 * (S < -1 * EPS) * S
        Pp = 1.0 * (S > 1 * EPS) * S
        self.Rp = Rp
        self.Pp = Pp

    def req_from_fluxes(self, fluxes):
        fluxes_neg = -1.0 * (fluxes < 0) * fluxes
        fluxes_pos = 1.0 * (fluxes > 0) * fluxes
        requests = np.dot(self.Rp, fluxes_pos) + np.dot(self.Pp, fluxes_neg)
        return requests
