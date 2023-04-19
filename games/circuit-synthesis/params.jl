using CUDA

gpu_available = CUDA.functional() 												# Check if GPU is available

Network = NetLib.ResNet 

netparams = NetLib.ResNetHP(
	num_filters=128,
	num_blocks=5,
	conv_kernel_size=(3, 3),
	num_policy_head_filters=32,
	num_value_head_filters=32,
	batch_norm_momentum=0.1)

  ## Learning parameters during the neural network gradient descent ##

  learning = LearningParams(
    use_gpu=gpu_available,                                                              # Use GPU
    use_position_averaging=true,                                               	# 
    samples_weighing_policy=LOG_WEIGHT,                                    # ??
    l2_regularization=1e-4,                                                     # L2 regulization
    optimiser=Adam(lr=1e-3),                                                    # Optimiser for the gradient descent
    batch_size=1024,                                                             # Size of the batch
    loss_computation_batch_size=1024,                                            # Batch size used to compute the loss between each epochs
    nonvalidity_penalty=1.,                                                     # Multiplicative constant of a loss term that corresponds to the average probability weight that the network puts on invalid actions
    min_checkpoints_per_epoch=1,                                                # ??
    max_batches_per_checkpoint=5_000,                                           # ??
    num_checkpoints=1)                                                          # ??


  ## Parameters of the games played against itself during training ##

  # Parameters of the games (simulations) :
  sim_selfplayed = SimParams(
    num_games=4_096,                                                # Number of games to be played
    num_workers=128,                                                            # Sim task to spawn
    batch_size=64,                                                              # Batch size of inference request 
    use_gpu=gpu_available,                                                              # Use gpu or not
    reset_every=8,                                                            # Reset the tree every _ games
    flip_probability=0.,                                                        # Probability o flipping the borad with symmetric transformation
    alternate_colors=false)                                                     # Not important

  # Parameters for the MCTS
  mcts_selfplayed = MctsParams(
    num_iters_per_turn=500,                               # Number of iteration for the MCTS (to get the statistics)
    cpuct=2.0,                                            # Exploration parameter in the UTC formula
	temperature=PLSchedule(1.0),                          # Temperature
    dirichlet_noise_ϵ=0.25,                               # Amount of dirchlet noise 
    dirichlet_noise_α=1.0)                                # Dirchlet noise parameter

  self_play = SelfPlayParams(sim=sim_selfplayed, mcts=mcts_selfplayed)

  ## Parameter during the competition of the old and new neural networks ##

  # Simulation parameters
  sim_arena = SimParams(
    num_games=128,                                          # Number of games
    num_workers=128,                                                            #
    batch_size=64,                                                             #
    use_gpu=gpu_available,                                                              #
    reset_every=1,                                                              #
    flip_probability=0.,                                                        #
    alternate_colors=true)                                                      #

  # MCTS parameters
  mcts_arena = self_play.mcts

  # Threshold for NN replacing
  threshold_arena = 0.0

  arena = ArenaParams(sim=sim_arena, mcts=mcts_arena, update_threshold=threshold_arena)

  ## Parameters ##

  params = Params(
    arena=arena,                                                                # Arena parameters
    self_play=self_play,                                                        # Selfplayed parameters
    learning=learning,                                                          # Learning parameters
    num_iters=5,                                                                  # Number of iteration
    memory_analysis=nothing,                                                    # Analysis of the memory buffer (cf Doc)
    ternary_rewards=false,                                                      # If reward {-1, 0, 1}
    use_symmetries=false,                                                       # Use symmetries of the board
	mem_buffer_size=PLSchedule(
	[      0,        5],
  	[400_000, 1_000_000]))                                         # Size schedule of the memory buffer, in terms of number of samples


  ## Benchmark

  benchmark_sim = SimParams(
    arena.sim;
    num_games=128,
    num_workers=64,
    batch_size=64)

  benchmark = [
    Benchmark.Single(
      Benchmark.Full(self_play.mcts),
      benchmark_sim),
    Benchmark.Single(
      Benchmark.NetworkOnly(),
      benchmark_sim)]

experiment = Experiment(
  "circuit-synthesis", GameSpec(), params, Network, netparams, benchmark)
