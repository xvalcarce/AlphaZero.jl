using CUDA

gpu_available = CUDA.functional() 												# Check if GPU is available

# OVERFIT : simple network, to be run with trivial game-set. This only to test the game, overfitting is the expected behaviour
# LEARN : complex network, requires GPU.
const LEARN = Dict("filters" => 128,
					 "blocks" => 5,
					 "policy_filters" => 32,
					 "value_filters" => 32,
					 "n_games" => 4_096,
					 "mcts_n_iter" => 500,
					 "temperature" => PLSchedule([0,20,30], [1.0,1.0,0.2]),
					 "batch_size" => 1_024,
					 "learning_rate" => 1e-3
					 )

const OVERFIT = Dict("filters" => 32,
					 "blocks" => 5,
					 "policy_filters" => 16,
					 "value_filters" => 16,
					 "n_games" => 256,
					 "mcts_n_iter" => 500,
					 "temperature" => PLSchedule([0,20,30], [1.0,1.0,0.2]),
					 "batch_size" => 128,
					 "learning_rate" => 1e-3
					 )

HParams = LEARN

## Neural network

# NN type
Network = NetLib.ResNet 

# NN Hyper params 
netparams = NetLib.ResNetHP(
	num_filters = HParams["filters"],
	num_blocks = HParams["blocks"],
	conv_kernel_size = (3, 3),
	num_policy_head_filters = HParams["policy_filters"],
	num_value_head_filters = HParams["value_filters"],
	batch_norm_momentum = 0.1)

## Self play parameters during exploration/learning phase

self_play = SelfPlayParams(
	# Parameters of the game simulations
	sim = SimParams(
		num_games = HParams["n_games"], 	# Number of games simulated per iterations
		num_workers = 128, 				# Number of workers (task) to spawn
		batch_size = 64, 				# Batch size of interference request
		use_gpu = gpu_available, 		# Whether to use the GPU or not
		reset_every = 1, 				# Number of game before the game tree (memory) is reset
		flip_probability = 0., 			# Probability to flip the board -> 0 when single player
		alternate_colors = false), 		# Single player so no
	# Parameters of the mcts
	mcts = MctsParams(
		num_iters_per_turn = HParams["mcts_n_iter"], 	# MCTS iterations
		cpuct = 2.0, 									# Exploration param in the UTC forumla
		prior_temperature = 1.0, 						#
		temperature = HParams["temperature"], 			# temperature
		dirichlet_noise_ϵ = 0.25, 						# Amount of dirichlet noise
		dirichlet_noise_α = 1.)) 						# Dirichlet noise amplitude

## Arena parameters : Neural network (and AZ) is benchmarked on these games 

arena = ArenaParams(
	sim = SimParams(
	    num_games = 128,
    	num_workers = 128,
    	batch_size = 64, 
    	use_gpu = gpu_available,
    	reset_every = 1,
    	flip_probability = 0.,
    	alternate_colors = true),
	mcts = MctsParams(
		self_play.mcts,
		temperature = ConstSchedule(0.2),
		dirichlet_noise_ϵ = 0.05),
	update_threshold = 0.05)

## Learning parameters : gradient descent, loss params, ...

learning = LearningParams(
	use_gpu = gpu_available,                                                              # Use GPU
    use_position_averaging = true,                                               	# 
    samples_weighing_policy = LOG_WEIGHT,                                    # ??
	batch_size = HParams["batch_size"],                                                             # Size of the batch
	loss_computation_batch_size = HParams["batch_size"],                                            # Batch size used to compute the loss between each epochs
    l2_regularization = 1e-4,                                                     # L2 regulization
    optimiser = Adam(lr=HParams["learning_rate"]),                                                    # Optimiser for the gradient descent
    nonvalidity_penalty = 1.,                                                     # Multiplicative constant of a loss term that corresponds to the average probability weight that the network puts on invalid actions
    min_checkpoints_per_epoch = 1,                                                # ??
    max_batches_per_checkpoint = 5_000,                                           # ??
    num_checkpoints = 1)                                                          # ??

## Parameters ##

params = Params(
	arena = arena,                                                                # Arena parameters
    self_play = self_play,                                                        # Selfplayed parameters
    learning = learning,                                                          # Learning parameters
    num_iters = 5,                                                                  # Number of iteration
    ternary_rewards = false,                                                      # If reward {-1, 0, 1}
    use_symmetries = false,                                                       # Use symmetries of the board
    memory_analysis = nothing,                                                    # Analysis of the memory buffer (cf Doc)
	mem_buffer_size = PLSchedule(
	[      0,        5],
  	[400_000, 1_000_000]))                                         # Size schedule of the memory buffer, in terms of number of samples

## Benchmark parameters

benchmark_sim = SimParams(
	arena.sim;
    num_games = 256,
    num_workers = 64,
    batch_size = 64)

benchmark = [
	Benchmark.Single(Benchmark.Full(arena.mcts), benchmark_sim), # mcts alone
    Benchmark.Single(Benchmark.NetworkOnly(;τ=1.0),benchmark_sim)] # network alone

## Experiment 
experiment = Experiment(
  "circuit-synthesis", GameSpec(), params, Network, netparams, benchmark)
