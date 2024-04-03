# Game parameters
const MAX_DEPTH = 40    	# Max depth of circuit to explore (excluding the target)
const MAX_TARGET_DEPTH = 20    # Max number of gate of the target circuit
const MODE = 3                # Number of modes
const HALF_TARGET_DEPTH = 10   # See WEIGHT
const WEIGHT = 0.25            # when rand() > WEIGHT generate a circuit of depth HALF_TARGET_DEPTH:MAX_TARGET_DEPTH, otherwise 1:HALF_TARGET_DEPTH

# Target gate set from which target circuits for training will be created
target_set = Dict("single_gate" => [H,Z,X,T,T',S,S'],
				"ctrl_gate" => [X,Z],
				"cctrl_gate" => [])

# Hardware gate set, which constitue gate set available to the compiler
hardware_set = Dict("single_gate" => [H,S,S',T,T'],
				"ctrl_gate" => [X],
				"cctrl_gate" => [])

target_set = hardware_set
