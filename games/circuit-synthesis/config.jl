# Game parameters
const MODE = 3                # Number of modes
const MAX_DEPTH = 40          # Max depth of circuit to explore (excluding the target)

const MIN_TARGET_DEPTH = 2    # Min number of gate of the target circuit
const MAX_TARGET_DEPTH = 40   # Max number of gate of the target circuit

# Circuit distribution params
#
# using a Normal distribution
const USE_NORMAL_DIST = true  # Whether to use a normal distribution for the circuit length
const MIN_MEAN_DEPTH = 2     # minimum mean of the normal distribution
const MAX_MEAN_DEPTH = 40     # maximum mean of the normal distribution 
const STD_DEV_DEPTH = 5       # standard deviation of the normal distribution
const ITC_MEAN_INCREMENT = 2  # increase the mean of the normal distribution every n training iterations (set to -1 for fix mean)
#
# Or a biased uniform distribution
const HALF_TARGET_DEPTH = 10   # See WEIGHT
const WEIGHT = 0.25            # when rand() > WEIGHT generate a circuit of depth HALF_TARGET_DEPTH:MAX_TARGET_DEPTH, otherwise MIN_TARGET_DEPTH:HALF_TARGET_DEPTH

# Map all the state equivalent to a global phase to the same matrix
const USE_GP_SYM = false

# Target gate set from which target circuits for training will be created
target_set = Dict("single_gate" => [H,Z,X,T,T',S,S'],
				"ctrl_gate" => [X,Z],
				"cctrl_gate" => [])

# Hardware gate set, which constitue gate set available to the compiler
hardware_set = Dict("single_gate" => [H,S,S',T,T'],
				"ctrl_gate" => [X],
				"cctrl_gate" => [])

# Setting the target_set to the hardware_set adds an extra stop condition
# namely, the depth of the compiled circuit can't exceed the depth of the current target circuit
target_set = hardware_set

# Benchmark parameters
A_DEPTH = 30 # testing on circuit of depth 30
audit_set = hardware_set # audit circuit from the hardware_set
A_MAX_DEPTH = 30 # maximum depth allowed
