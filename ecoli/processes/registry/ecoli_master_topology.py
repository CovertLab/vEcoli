from vivarium.core.registry import Registry

from ecoli.processes.tf_binding import TfBinding
from ecoli.processes.transcript_initiation import TranscriptInitiation
from ecoli.processes.transcript_elongation import TranscriptElongation
from ecoli.processes.rna_degradation import RnaDegradation
from ecoli.processes.polypeptide_initiation import PolypeptideInitiation
from ecoli.processes.polypeptide_elongation import PolypeptideElongation
from ecoli.processes.complexation import Complexation
from ecoli.processes.two_component_system import TwoComponentSystem
from ecoli.processes.equilibrium import Equilibrium
from ecoli.processes.protein_degradation import ProteinDegradation
from ecoli.processes.metabolism import Metabolism
from ecoli.processes.chromosome_replication import ChromosomeReplication
from ecoli.processes.mass import Mass
from ecoli.processes.exchange_stub import Exchange
from ecoli.processes.listeners.mass_listener import MassListener


#: Maps process names to topology
topology_registry = Registry()
topology_registry.register(
    TfBinding.name,
    {
        "promoters": ("unique", "promoter"),
        "active_tfs": ("bulk",),
        "inactive_tfs": ("bulk",),
        "listeners": ("listeners",)
    })
topology_registry.register(
    TranscriptInitiation.name,
    {
        "environment": ("environment"),
        "full_chromosomes": ("unique", "full_chromosome"),
        "RNAs": ("unique", "RNA"),
        "active_RNAPs": ("unique", "active_RNAP"),
        "promoters": ("unique", "promoter"),
        "molecules": ("bulk"),
        "listeners": ("listeners")
    })
topology_registry.register(
    TranscriptInitiation.name,
    {
        "environment": ("environment"),
        "full_chromosomes": ("unique", "full_chromosome"),
        "RNAs": ("unique", "RNA"),
        "active_RNAPs": ("unique", "active_RNAP"),
        "promoters": ("unique", "promoter"),
        "molecules": ("bulk"),
        "listeners": ("listeners")
    })
topology_registry.register(
    TranscriptElongation.name,
    {
        "environment": ("environment"),
        "RNAs": ("unique", "RNA"),
        "active_RNAPs": ("unique", "active_RNAP"),
        "molecules": ("bulk"),
        "bulk_RNAs": ("bulk"),
        "ntps": ("bulk"),
        "listeners": ("listeners")
    })
topology_registry.register(
    RnaDegradation.name,
    {
        "charged_trna": ("bulk"),
        "bulk_RNAs": ("bulk"),
        "nmps": ("bulk"),
        "fragmentMetabolites": ("bulk"),
        "fragmentBases": ("bulk"),
        "endoRnases": ("bulk"),
        "exoRnases": ("bulk"),
        "subunits": ("bulk"),
        "molecules": ("bulk"),
        "RNAs": ("unique", "RNA"),
        "active_ribosome": ("unique", "active_ribosome"),
        "listeners": ("listeners")
    })
topology_registry.register(
    PolypeptideInitiation.name,
    {
        "environment": ("environment"),
        "listeners": ("listeners"),
        "active_ribosome": ("unique", "active_ribosome"),
        "RNA": ("unique", "RNA"),
        "subunits": ("bulk")
    })
topology_registry.register(
    PolypeptideElongation.name,
    {
        "environment": ("environment"),
        "listeners": ("listeners"),
        "active_ribosome": ("unique", "active_ribosome"),
        "molecules": ("bulk"),
        "monomers": ("bulk"),
        "amino_acids": ("bulk"),
        "ppgpp_reaction_metabolites": ("bulk"),
        "uncharged_trna": ("bulk"),
        "charged_trna": ("bulk"),
        "charging_molecules": ("bulk"),
        "synthetases": ("bulk"),
        "subunits": ("bulk"),
        "polypeptide_elongation": ("process_state", "polypeptide_elongation")
    })
topology_registry.register(
    Complexation.name,
    {
        "molecules": ("bulk")
    })
topology_registry.register(
    TwoComponentSystem.name,
    {
        "listeners": ("listeners"),
        "molecules": ("bulk")
    })
topology_registry.register(
    Equilibrium.name,
    {
        "listeners": ("listeners"),
        "molecules": ("bulk")
    })
topology_registry.register(
    ProteinDegradation.name,
    {
        "metabolites": ("bulk"),
        "proteins": ("bulk")
    })
topology_registry.register(
    Metabolism.name,
    {
        "metabolites": ("bulk"),
        "catalysts": ("bulk"),
        "kinetics_enzymes": ("bulk"),
        "kinetics_substrates": ("bulk"),
        "amino_acids": ("bulk"),
        "listeners": ("listeners"),
        "environment": ("environment"),
        "polypeptide_elongation": ("process_state", "polypeptide_elongation")
    })
topology_registry.register(
    ChromosomeReplication.name,
    {
        "replisome_trimers": ("bulk"),
        "replisome_monomers": ("bulk"),
        "dntps": ("bulk"),
        "ppi": ("bulk"),
        "active_replisomes": ("unique", "active_replisome"),
        "oriCs": ("unique", "oriC"),
        "chromosome_domains": ("unique", "chromosome_domain"),
        "full_chromosomes": ("unique", "full_chromosome"),
        "listeners": ("listeners"),
        "environment": ("environment")
    })
topology_registry.register(
    MassListener.name,
    {
        "bulk": ("bulk"),
        "unique": ("unique"),
        "listeners": ("listeners")
    })
