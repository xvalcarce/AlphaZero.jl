using Yao 
using StaticArrays
using LinearAlgebra
using Distributions

import AlphaZero.GI

# Extra Yao gate definition
@const_gate S = Diagonal{ComplexF64}([1,1im])
@const_gate Sdag = Diagonal{ComplexF64}([1,-1im])
Base.adjoint(::SGate) = Sdag
Base.adjoint(::SdagGate) = S

# Import config file with game paramters
include("./config.jl")

# Import helper functions
include("./helper.jl")

# Define depth distribution for random circuit sampling
# using ref (pointer) so variable can be updated using GI.update_gspec
const MEAN = Ref(MIN_MEAN_DEPTH)
tndist(mean::Int) = Truncated(Normal(mean, STD_DEV_DEPTH), MIN_TARGET_DEPTH, MAX_TARGET_DEPTH)
const DIST = USE_NORMAL_DIST ? Ref(tndist(MEAN[])) : Ref(BiasUniform())

#Define QCir type and useful functions
include("./qcir.jl")

const DIM = 2^MODE             # Size of the matrix representing a circuit
const ELEM = DIM^2             # Number of complex element of a desntiy matrix 
# Target gate set
const T_GATESET = buildGateSet(MODE, target_set)
const T_REDUNDANCY = buildRedudancyDict(T_GATESET)
# Hardware (compiler) gate 
const H_GATESET = buildGateSet(MODE, hardware_set)
const H_GATESET_L = length(H_GATESET) # Length of the gateset
const H_REDUNDANCY = buildRedudancyDict(H_GATESET)
# check if gateset are the same
const SAME_GATESET = (hardware_set == target_set)

const MAT_ID = SparseMatrixCSC{ComplexF64}(I,DIM,DIM) # SparseMatrix Identity
const HASH_ID =  hash(mapCanonical(MAT_ID)) # Hash of "Id" for fidelity (used as a reward)

# GameSpec 
struct GameSpec <: GI.AbstractGameSpec end

# Define the environement
mutable struct GameEnv <: GI.AbstractGameEnv
	circuit::QCir{Hardware} # Current circuit
	target::QCir{Target} # Target circuit
	adj_m_target::SparseMatrixCSC # Adjoint of mat target for speedup
	reward::Bool
end

GI.spec(::GameEnv) = GameSpec()

# Single player game
GI.two_players(::GameSpec) = false
GI.white_playing(::GameEnv) = true

# Reward
if USE_GP_SYM
	reward(u::QCir,t::SparseMatrixCSC) = HASH_ID == hash(mapCanonical(t*u.m))
else
	reward(u::QCir,t::SparseMatrixCSC) = MAT_ID == mapCanonical(t*u.m)
end
GI.white_reward(game::GameEnv) :: Float64 = game.reward ? 1. : 0.

# Init with random target circuit and empty cir
function GI.init(::GameSpec)
	c = QCir{Hardware}()
	t = rand(DIST[],Target)
	atm = adjoint(t.m)
	return GameEnv(c,t,atm,false)
end

# Init from state (target and circuit)
function GI.init(::GameSpec, state)
	c = QCir{Hardware}(copy(state.circuit))
	t = QCir{Target}(copy(state.target))
	atm = sparse(adjoint(t.m))
	r = reward(c,atm)
	return GameEnv(c,t,atm,r)
end

# Current state is target + circuit (target is needed for vectorize_state)
GI.current_state(game::GameEnv) = (target=copy(game.target.c), circuit=copy(game.circuit.c))

# Set the state according to nametupled, target shouldn't change tho (otherwise time overhead)
function GI.set_state!(game::GameEnv, state)
	game.circuit = QCir{Hardware}(copy(state.circuit))
	if game.target.c != state.target
		@debug "Diff target" game.target.c state.target
		game.target = QCir{Target}(state.target)
		game.adj_m_target = adjoint(game.target.m)
	end
	return 
end

# Action space
GI.actions(::GameSpec) = UInt8(1):UInt8(H_GATESET_L)

# Mask for valid actions
function GI.actions_mask(game::GameEnv) :: Vector{Bool}
	u = trues(H_GATESET_L)
	length(game.circuit.c) == 0 && return u
	for a in eachindex(u)
		@inbounds u[a] = !isRedundant(game.circuit.c, UInt8(a), H_REDUNDANCY)
	end
	return u
end

# Clone a game env
function GI.clone(game::GameEnv)
	circuit = copy(game.circuit)
	target = copy(game.target)
	return GameEnv(circuit, target, adjoint(target.m), game.reward)
end

# Action effect
function GI.play!(game::GameEnv, action)
	game.circuit = game.circuit(action)
	game.reward = reward(game.circuit,game.adj_m_target)
	return
end

# Termination conditions
GI.game_terminated(game::GameEnv) = game.reward || length(game.circuit.c) ≥ MAX_DEPTH || (SAME_GATESET && length(game.circuit.c) ≥ length(game.target.c)) 

# Vectorize repr of a state, fed to the NN
function GI.vectorize_state(::GameSpec, state)
	c = QCir{Hardware}(state.circuit).m
	t = adjoint(QCir{Target}(state.target).m)
	m = mapCanonical(t*c)
	vs = Float32[f(m[i,j]) for i in 1:DIM, j in 1:DIM, f in [real,imag]]
	return vs
end

# Read state from stdin
function GI.read_state(::GameSpec)
	try
		state = []
		for i in 1:2
			input = readline()
			input = split(input," ")
			c = map(x -> parse(UInt8,x), input)
			push!(state,c)
		end
		return (target=state[1],circuit=state[2])
	catch e
		return nothing
	end
end

function GI.update_gspec(::GameSpec,itc::Int)
	@info ITC_MEAN_INCREMENT
	@info DIST[]
	@info MEAN[]
	if USE_NORMAL_DIST && (itc+1) % ITC_MEAN_INCREMENT == 0 && MEAN[] < MAX_MEAN_DEPTH 
		MEAN[] = min(MEAN[] + 1, MAX_MEAN_DEPTH)
		DIST[] = tndist(MEAN[])
		@info "Normal distribution mean incremented to $(MEAN[])."
	end
	return
end

## Additional methods
# Non mandatory methods that can be useful for the game representation

# For minmax player
GI.heuristic_value(::GameEnv) = 0.

function GI.render(game::GameEnv)
	println("TARGET: ")
	print(game.target)
	println()
	println("CIRCUIT: ")
	print(game.circuit)
end

function GI.action_string(gs::GameSpec, a)
	idx = findfirst(==(a), GI.actions(gs))
	return isnothing(idx) ? "?" : H_GATESET[idx].name
end

function GI.parse_action(gs::GameSpec, s)
	idx = findfirst(==(s), [hg.name for hg in H_GATESET])
	return isnothing(idx) ? nothing : GI.actions(gs)[idx]
end
