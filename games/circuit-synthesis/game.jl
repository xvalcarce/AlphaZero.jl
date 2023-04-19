using CommonRLInterface
using Yao 
using StaticArrays
using LinearAlgebra
include("./helper.jl")

const RL = CommonRLInterface

# Game parameters
const MAX_DEPTH = 20    	   # Max depth of circuit to explore (excluding the target)
const MAX_TARGET_DEPTH = 10    # Max number of gate of the target circuit
const MODE = 3                 # Number of modes
const DIM = 2^MODE              # Size of the matrix representing a circuit
const ELEM = DIM^2              # Number of complex element of a desntiy matrix 
#const TARGET = chain(MODE, put(1=>X), put(2=>H), control(2, 1=>X)) # Target Circuit
#const MAT_TARGET = Matrix(mat(TARGET)) # Matrix rep of the target randomCircuit
const S = shift(π/2)           # Shift gate
const GATE = [H, T, T', S, S'] # Possible Gates 
const CONTROL = [X]            # Possible Control Gates

const GATESET = gateset(MODE, GATE, CONTROL)
const GATESET_NAME = gateset_name(MODE, GATE, CONTROL)

# Define the environement
mutable struct World <: AbstractEnv
	state::Matrix
	circuit::ChainBlock
	depth::UInt8
end

# Constructor
World(state, circuit) = World(state, circuit, length(circuit))
function World()
	t = randomCircuit(MODE,GATESET)
	return World(Matrix(mat(t)),t,0)
end

fidelity(u::AbstractMatrix) = round(abs(tr(u)) / DIM, digits=6)
fidelity(env::World) = fidelity(env.state)

## Default methods of CommonRLInterface
# Here are the methods for CommonRLInterface, other methods are needed for AlphaZero

RL.actions(env::World) = UInt8(1):UInt8(length(GATESET))
RL.observe(env::World) = env.state
RL.terminated(env::World) = fidelity(env) == 1.0 || env.depth > MAX_DEPTH

# Interaction function
function RL.act!(env::World, action)
	env.depth += 1													# +1 to the current depth
	actionChain = GATESET[action]									# Get gate that corresponds to the action
	push!(env.circuit, actionChain)									# Update the circuit
	env.state = Matrix(mat(env.circuit))
	fid = fidelity(env.state)
	if fid == 1.0
		return +1
	else
		return 0
	end
end

# Reset the game
function RL.reset!(env::World)
	# Generate new target
	t = randomCircuit(MODE,GATESET)
	env.state = Matrix(mat(t))
	env.circuit = t
	env.depth = 0
	return nothing
end

## Mandatory methods for AlphaZero
# Additional methods used by the AlphaZero package

# Clone the current environement

# Current state
function to_vec(env::World)
	v = vec(env.state)
	v = vcat(real(v), imag(v))
	return Vector{Float32}(v)
end

function to_mat(v::Vector)
	@assert length(v) == 2*ELEM
	m = Matrix{ComplexF64}(undef, DIM, DIM)
	k = 1
	@inbounds for i in 1:DIM
		for j in 1:DIM
			m[j,i] = v[k]+im*v[ELEM+k]
			k += 1
		end
	end
	m
end

@provide RL.clone(env::World) = World(copy(env.state),copy(env.circuit),env.depth)
@provide RL.state(env::World) = env.state
@provide RL.setstate!(env::World, s::Matrix{ComplexF64}) = (env.state = s)
@provide RL.valid_action_mask(env::World) = BitVector([1 for i in 1:length(GATESET)])
@provide RL.player(env::World) = 1
@provide RL.players(env::World) = [1]

## Additional methods
# Non mandatory methods that can be useful for the game representation

GI.heuristic_value(::World) = 0.

function GI.render(env::World)
	print(env.circuit)
end

function GI.vectorize_state(env::World, state)
	return Float32[
		f(state[i,j])
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
