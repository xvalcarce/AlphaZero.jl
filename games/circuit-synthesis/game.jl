using CommonRLInterface
using Yao 
using StaticArrays
using LinearAlgebra

const RL = CommonRLInterface

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

# Define the environement
mutable struct World <: AbstractEnv
	circuit::QCir # Current circuit
	target::QCir # Target circuit
	adj_m_target::SparseMatrixCSC # Adjoint of mat target for speedup
end

# Constructor
function World()
	c = QCir()
	t = randQCir()
	atm = adjoint(t.m)
	return World(c,t,atm)
end

# Reward functions
reward(u::QCir,t::SparseMatrixCSC) = HASH_ID == hash(mapCanonical(t*u.m))
reward(env::World) = reward(env.circuit,env.adj_m_target)

## Default methods of CommonRLInterface
# Here are the methods for CommonRLInterface, other methods are needed for AlphaZero
RL.actions(::World) = UInt8(1):UInt8(L_GATESET)
RL.observe(env::World) = mapCanonical(env.adj_m_target*env.circuit.m)
RL.terminated(env::World) = reward(env) || length(env.circuit.c) > MAX_DEPTH

function RL.valid_action_mask(env::World)
	u = BitVector(undef, L_GATESET)
	@inbounds for i in 1:L_GATESET
		u[i] = !isRedundant(env.circuit,UInt8(i))
	end
	return u
end

# Interaction function
function RL.act!(env::World, action)
	# update the circuit
	env.circuit = env.circuit(action)
	# compute reward
	if reward(env)
		return +1
	else
		return 0
	end
end

# Reset the game
function RL.reset!(env::World)
	# Reset circuit to an empty one
	env.circuit = QCir()
	# Generate new target
	t = randQCir()
	env.target = t
	env.adj_m_target = adjoint(t.m)
	return nothing
end

@provide RL.clone(env::World) = World(copy(env.circuit),copy(env.target),copy(env.adj_m_target))
@provide RL.state(env::World) = copy(env.circuit.c)
@provide RL.setstate!(env::World, c::Vector{UInt8}) = (env.circuit = QCir(copy(c)))
@provide RL.player(::World) = 1
@provide RL.players(::World) = [1]

# Vectorize repr of a state, fed to the NN
function GI.vectorize_state(env::World, state::Vector{UInt8})
	m = mapCanonical(env.adj_m_target*QCir(state).m)
	vs = Float32[f(m[i,j]) for i in 1:DIM, j in 1:DIM, f in [real,imag]]
	return vs
end

## Additional methods
# Non mandatory methods that can be useful for the game representation

# For minmax player
GI.heuristic_value(::World) = 0.

function GI.render(env::World)
	println("TARGET: ")
	print(env.target)
	println()
	println("CIRCUIT: ")
	print(env.circuit)
end

function GI.action_string(env::World, a)
	idx = findfirst(==(a), RL.actions(env))
	return isnothing(idx) ? "?" : GATESET_NAME[idx]
end

function GI.parse_action(env::World, s)
	idx = findfirst(==(s), GATESET_NAME)
	return isnothing(idx) ? nothing : RL.actions(env)[idx]
end

# Generate GameSpec using RL wrapper
GameSpec() = CommonRLInterfaceWrapper.Spec(World())
