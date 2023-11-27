using Yao 
using StaticArrays
using LinearAlgebra

import AlphaZero.GI

# Game parameters
const MAX_DEPTH = 20    	   # Max depth of circuit to explore (excluding the target)
const MAX_TARGET_DEPTH = 10    # Max number of gate of the target circuit
const MODE = 2                 # Number of modes
const DIM = 2^MODE             # Size of the matrix representing a circuit
const ELEM = DIM^2             # Number of complex element of a desntiy matrix 

include("./helper.jl")
# Gateset considered
@const_gate S = Diagonal{ComplexF64}([1,1im])
@const_gate Sdag = Diagonal{ComplexF64}([1,-1im])
Base.adjoint(::SGate) = Sdag
Base.adjoint(::SdagGate) = S
const HERM_GATE = [H]
const GATE = [H, T, T', S, S'] # Possible Gates 
const CONTROL = [X]            # Possible Control Gates
const GATESET,GATESET_NAME,GATESET_REF = gateset(MODE, GATE, CONTROL) # All gates: Yao repr, name, references
const M_GATESET = [sparse(mat(g(MODE))) for g in GATESET] # Pre-generating matrix repr of gateset
const L_GATESET = length(GATESET) # Length of the gateset
const HASH_ID =  hash(mapCanonical(SparseMatrixCSC{ComplexF64}(I,DIM,DIM))) # Hash of "Id" for fidelity (used as a reward)

#Define QCir and some helper function
include("./qcir.jl")

# GameSpec 
struct GameSpec <: GI.AbstractGameSpec end

# Define the environement
mutable struct GameEnv <: GI.AbstractGameEnv
	circuit::QCir # Current circuit
	target::QCir # Target circuit
	adj_m_target::SparseMatrixCSC # Adjoint of mat target for speedup
	reward::Bool
end

GI.spec(::GameEnv) = GameSpec()

# Single player game
GI.two_players(::GameSpec) = false
GI.white_playing(::GameEnv) = true

# Reward
reward(u::QCir,t::SparseMatrixCSC) = HASH_ID == hash(mapCanonical(t*u.m))
GI.white_reward(game::GameEnv) = game.reward ? 1 : 0

# Init with random target circuit and empty cir
function GI.init(::GameSpec)
	c = QCir()
	t = randQCir()
	atm = adjoint(t.m)
	return GameEnv(c,t,atm,false)
end

function GI.init(::GameSpec, state)
	c = QCir(copy(state.circuit))
	t = QCir(copy(state.target))
	atm = sparse(adjoint(t.m))
	r = reward(c,atm)
	return GameEnv(c,t,atm,r)
end

# Current state is target + circuit (target is needed for vectorize_state)
GI.current_state(game::GameEnv) = (target=game.target.c, circuit=copy(game.circuit.c))

# Set the state according to nametupled, target shouldn't change tho (otherwise time overhead)
function GI.set_state!(game::GameEnv, state)
	game.circuit = QCir(state.circuit)
	if game.target.c != state.target
		@debug "Diff target" game.target.c state.target
		game.target = QCir(state.target)
		game.adj_m_target = adjoint(game.target.m)
	end
	return 
end

# Action space
GI.actions(::GameSpec) = UInt8(1):UInt8(L_GATESET)

# Mask for valid actions
function GI.actions_mask(game::GameEnv)
	u = BitVector(undef, L_GATESET)
	@inbounds for i in 1:L_GATESET
		u[i] = !isRedundant(game.circuit,UInt8(i))
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
GI.game_terminated(game::GameEnv) = game.reward || length(game.circuit.c) ≥ MAX_DEPTH

# Vectorize repr of a state, fed to the NN
function GI.vectorize_state(::GameSpec, state)
	c = QCir(state.circuit).m
	t = adjoint(QCir(state.target).m)
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
	return isnothing(idx) ? "?" : GATESET_NAME[idx]
end

function GI.parse_action(gs::GameSpec, s)
	idx = findfirst(==(s), GATESET_NAME)
	return isnothing(idx) ? nothing : GI.actions(gs)[idx]
end
