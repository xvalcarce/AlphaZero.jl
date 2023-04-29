using CommonRLInterface
using Yao 
using StaticArrays
using LinearAlgebra
include("./helper.jl")

const RL = CommonRLInterface
Base.hash(c::ChainBlock) = hash(mat(c)) # Hacky but works for mcts keys

# Game parameters
const MAX_DEPTH = 20    	   # Max depth of circuit to explore (excluding the target)
const MAX_TARGET_DEPTH = 10    # Max number of gate of the target circuit
const MODE = 2                 # Number of modes
const DIM = 2^MODE              # Size of the matrix representing a circuit
const ELEM = DIM^2              # Number of complex element of a desntiy matrix 

# Gateset considered
@const_gate S = Diagonal{ComplexF64}([1,1im])
@const_gate Sdag = Diagonal{ComplexF64}([1,-1im])
Base.adjoint(::SGate) = Sdag
Base.adjoint(::SdagGate) = S
const HERM_GATE = [H]
const GATE = [H, T, T', S, S'] # Possible Gates 
const CONTROL = [X]            # Possible Control Gates
const GATESET = gateset(MODE, GATE, CONTROL)
const L_GATESET = length(GATESET)
const GATESET_NAME = gateset_name(MODE, GATE, CONTROL)

# Define the environement
mutable struct World <: AbstractEnv
	circuit::ChainBlock # Current state
	target::ChainBlock  # Target circuit
	target_mat_dag::Matrix # For speedup
end

# Constructoer
function World()
	t = randomCircuit(MODE,GATESET)
	return World(chain(MODE),t,adjoint(mat(t)))
end

# Fidelity (used as a reward)
fidelity(u::ChainBlock,t::Matrix) = round(abs(tr(t*mat(u))) / DIM, digits=6)
fidelity(env::World) = fidelity(env.circuit,env.target_mat_dag)

## Default methods of CommonRLInterface
# Here are the methods for CommonRLInterface, other methods are needed for AlphaZero

RL.actions(env::World) = UInt8(1):UInt8(L_GATESET)
RL.observe(env::World) = copy(env.circuit)
RL.terminated(env::World) = fidelity(env.circuit,env.target_mat_dag) == 1.0 || length(env.circuit) > MAX_DEPTH

function RL.valid_action_mask(env::World)
	u = BitVector(undef, L_GATESET)
	@inbounds for i in 1:L_GATESET
		u[i] = !isRedundant(env.circuit,GATESET[i])
	end
	return u
end

# Interaction function
function RL.act!(env::World, action)
	actionChain = GATESET[action]									# Get gate that corresponds to the action
	push!(env.circuit, actionChain)									# Update the circuit
	fid = fidelity(env.circuit,env.target_mat_dag)
	if fid == 1.0
		return +1
	else
		return 0
	end
end

# Reset the game
function RL.reset!(env::World)
	# Generate new target
	env.circuit = chain(MODE)
	t = randomCircuit(MODE,GATESET)
	env.target = t
	env.target_mat_dag = adjoint(mat(t))
	return nothing
end

@provide RL.clone(env::World) = World(copy(env.circuit),copy(env.target),copy(env.target_mat_dag))
@provide RL.state(env::World) = copy(env.circuit)
@provide RL.setstate!(env::World, s::ChainBlock) = (env.circuit = copy(s)) # Copy for persistence
@provide RL.player(env::World) = 1
@provide RL.players(env::World) = [1]

## Additional methods
# Non mandatory methods that can be useful for the game representation

GI.heuristic_value(::World) = 0.

function GI.render(env::World)
	print(env.target)
	print(env.circuit)
end

function GI.vectorize_state(env::World, state::ChainBlock)
	m = env.target_mat_dag*mat(state)
	return Float32[
		f(m[i,j])
		for i in 1:DIM,
			j in 1:DIM,
			f in [real,imag]]
end

function GI.action_string(env::World, a)
	idx = findfirst(==(a), RL.actions(env))
	return isnothing(idx) ? "?" : GATESET_NAME[idx]
end

function GI.parse_action(env::World, s)
	idx = findfirst(==(s), GATESET_NAME)
	return isnothing(idx) ? nothing : RL.actions(env)[idx]
end

GameSpec() = CommonRLInterfaceWrapper.Spec(World())
