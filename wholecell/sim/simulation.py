MAX_TIME_STEP = 2

DEFAULT_SIMULATION_KWARGS = dict(
	timeline = '0 minimal',
	boundary_reactions = [],
	seed = 0,
	lengthSec = 3*60*60, # 3 hours max
	initialTime = 0.,
	jit = True,
	massDistribution = True,
	dPeriodDivision = False,
	growthRateNoise = False,
	translationSupply = True,
	trna_charging = True,
	ppgpp_regulation = False,
	superhelical_density = False,
	recycle_stalled_elongation = False,
	mechanistic_replisome = True,
	mechanistic_aa_supply = False,
	trna_attenuation = False,
	timeStepSafetyFraction = 1.3,
	maxTimeStep = MAX_TIME_STEP,
	updateTimeStepFreq = 5,
	logToShell = True,
	logToDisk = False,
	outputDir = None,
	logToDiskEvery = 1,
	simData = None,
	inheritedStatePath = None,
	variable_elongation_translation = False,
	variable_elongation_transcription = False,
	raise_on_time_limit = False,
	to_report = {
		# Iterable of molecule names
		'bulk_molecules': (),
		'unique_molecules': (),
		# Tuples of (listener_name, listener_attribute) such that the
		# desired value is
		# self.listeners[listener_name].listener_attribute
		'listeners': (),
	},
	cell_id = None,
)