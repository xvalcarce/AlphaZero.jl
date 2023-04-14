using CommonRLInterface
using Yao 
using StaticArrays
using LinearAlgebra

const RL = CommonRLInterface

## Environements

const MAX_DEPTH::UInt8 = 20 # Max depth of circuit to explore
const MODE = 2       # Number of modes
const DIM = 2MODE    # Size of the matrix representing a circuit
const TARGET = chain(MODE, put(1=>X), put(2=>H), control(2, 1=>X)) # Target Circuit
const MAT_TARGET = Matrix(mat(TARGET))
const GATE = [X, Z, H] # Possible Gates 
const CONTROL = [X]       # Possible Control Gates

function gateset(modes::Int, gateset::Vector, ctrl_set::Vector)
	# gate constructor for gates on ALL qubits
	gates = []
	for g in gateset
		for bit in 1:modes
			push!(gates, put(bit=>g))
		end
	end
	# add all controlled gates
	for g in ctrl_set
		for t in 1:modes
			for c in 1:modes
				if t != c
					push!(gates, control(c, t=>g))
				end
			end
		end
	end
	return gates
end

const GATESET = gateset(MODE, GATE, CONTROL)

# Define the environement
mutable struct World <: AbstractEnv
	state::Matrix
	circuit::ChainBlock
	depth::UInt8
end

# Constructor
#
function World(state, circuit)
	return World(state, circuit, length(circuit))
end
World() = World(MAT_TARGET,TARGET,0)

fidelity(u::AbstractMatrix) = round(abs(tr(u)) / DIM, digits=8)
fidelity(env::World) = fidelity(env.state)

## Default methods of CommonRLInterface
# Here are the methods for CommonRLInterface, other methods are needed for AlphaZero

# Allowed actions
RL.actions(env::World) = UInt8(1):UInt8(length(GATESET))

# Current state
RL.observe(env::World) = env.state

# End condition
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
		return 0.0													# Return the reward
	end
end

# Reset the game
function RL.reset!(env::World)
	env.state = MAT_TARGET
	env.circuit = TARGET
	env.depth = 0
end

## Mandatory methods for AlphaZero
# Additional methods used by the AlphaZero package

# Clone the current environement
@provide RL.clone(env::World) = World(env.state,env.circuit,env.depth)

# Current state
function to_vector(env::World)
	v = vec(env.state)
	v = vcat(real(v), imag(v))
	return v
end

@provide RL.state(env::World) = to_vector(env)

# Current state
function to_mat(v::Vector)
	@assert length(v) == 2*(DIM^2)
	m = Matrix{ComplexF64}(undef, DIM, DIM)
	k = 1
	@inbounds for i in 1:DIM
		for j in 1:DIM
			m[j,i] = v[k]+im*v[16+k]
			k += 1
		end
	end
	m
end

@provide RL.setstate!(env::World, s::Vector) = (env.state = to_mat(s))

# Mask corresponding to the valid actions of the current state
@provide RL.valid_action_mask(env::World) = BitVector([1 for i in 1:length(GATESET)])

# Number of players
@provide RL.player(env::World) = 1

# Vector of the players
@provide RL.players(env::World) = [1]

## Additional methods
# Non mandatory methods that can be useful for the game representation

function GI.render(env::World)
	print(env.state)
end

function GI.vectorize_state(env::World, state)
	return state
end

function GI.action_string(env::World, a)
	idx = findfirst(==(a), RL.actions(env))
	return isnothing(idx) ? "?" : GATESET[idx]
end

function GI.parse_action(env::World, s)
	idx = findfirst(==(s), GATESET)
	return isnothing(idx) ? nothing : RL.actions(env)[idx]
end

GameSpec() = CommonRLInterfaceWrapper.Spec(World())
