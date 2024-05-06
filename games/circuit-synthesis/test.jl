using Yao 
using StaticArrays
using SparseArrays
using LinearAlgebra

import AlphaZero.GI

struct Audit <: Architecture end

# Custom gate 
mutable struct UGate <: PrimitiveBlock{1}
	m::Union{Matrix,SparseMatrixCSC,Diagonal}
end

Yao.mat(::Type{T}, gate::UGate) where T = gate.m

# To test a specific unitary
struct UnitaryGate <: AbstractGate
	name::String
	mat::Union{SparseMatrixCSC,Diagonal}
	function UnitaryGate(name,mat)
		@assert size(mat) == (DIM, DIM)
		new(name, mat)
	end
end

UnitaryGate(mat::Union{SparseMatrixCSC,Diagonal}) = UnitaryGate("U", mat)
prettyprint(g::UnitaryGate) = println("$(g.name)")

# Example of custom gate set with a custom gate
# A_DEPTH = 10
# U = UGate(sparse([1 0; 0 exp(1im*π/6)]))
# audit_set = Dict("single_gate" => [U],
# 				"ctrl_gate" => [],
# 				"cctrl_gate" => [])
# A_GATESET = buildGateSet(MODE, audit_set)

# Example using single target gate
# A_DEPTH = 1
# A_MAX_DEPTH = MAX_DEPTH
# u = UnitaryGate(sparse([1,2,3,8,5,6,7,4],[1,2,3,4,5,6,7,8],[1.,1.,1.,1.,1.,1.,1.,1+0*1im]))
# A_GATESET = Vector{Any}([u]) 
# A_REDUNDANCY = Dict()

# Using value set in config.jl
A_GATESET = buildGateSet(MODE, audit_set)
A_REDUNDANCY = buildRedudancyDict(A_GATESET)

# Redundancy dict
redundancy(::Type{Audit}) = A_REDUNDANCY

# Gateset
gateset(::Type{Audit}) = A_GATESET

# GameSpec 
struct GameSpecAudit <: GI.AbstractGameSpec end

# Define the environement
mutable struct GameEnvAudit <: GI.AbstractGameEnv
	circuit::QCir{Hardware} # Current circuit
	target::QCir{Audit} # Target circuit
	adj_m_target::SparseMatrixCSC # Adjoint of mat target for speedup
	reward::Bool
end

GI.spec(::GameEnvAudit) = GameSpecAudit()

# Single player game
GI.two_players(::GameSpecAudit) = false
GI.white_playing(::GameEnvAudit) = true

GI.white_reward(game::GameEnvAudit) = game.reward ? 1 : 0

# Init with random target circuit and empty cir
function GI.init(::GameSpecAudit)
	c = QCir{Hardware}()
	t = rand(Audit,A_DEPTH)
	atm = ANCILLA_ARCH ? t.m[mask_i,mask_j] : t.m
	atm = sparse(adjoint(atm))
	return GameEnvAudit(c,t,atm,false)
end

# Init from state (target and circuit)
function GI.init(::GameSpecAudit, state)
	c = QCir{Hardware}(copy(state.circuit))
	t = QCir{Audit}(copy(state.target))
	atm = ANCILLA_ARCH ? t.m[mask_i,mask_j] : t.m
	atm = sparse(adjoint(atm))
	r = reward(c,atm)
	return GameEnvAudit(c,t,atm,r)
end

# Current state is target + circuit (target is needed for vectorize_state)
GI.current_state(game::GameEnvAudit) = (target=game.target.c, circuit=copy(game.circuit.c))

# Set the state according to nametupled, target shouldn't change tho (otherwise time overhead)
function GI.set_state!(game::GameEnvAudit, state)
	game.circuit = QCir{Hardware}(copy(state.circuit))
	if game.target.c != state.target
		@debug "Diff target" game.target.c state.target
		game.target = QCir{Audit}(state.target)
		atm = ANCILLA_ARCH ? game.target.m[mask_i,mask_j] : game.target.m
		atm = sparse(adjoint(atm))
		game.adj_m_target = atm
	end
	return 
end

# Action space
GI.actions(::GameSpecAudit) = UInt8(1):UInt8(H_GATESET_L)

# Mask for valid actions
function GI.actions_mask(game::GameEnvAudit) :: Vector{Bool}
	u = trues(length(H_GATESET))
	length(game.circuit.c) == 0 && return u
	for a in eachindex(u)
		@inbounds u[a] = !isRedundant(game.circuit.c, UInt8(a), H_REDUNDANCY)
	end
	return u
end

# Clone a game env
function GI.clone(game::GameEnvAudit)
	circuit = copy(game.circuit)
	target = copy(game.target)
	atm = ANCILLA_ARCH ? target.m[mask_i,mask_j] : target.m
	atm = sparse(adjoint(atm))
	return GameEnvAudit(circuit, target, atm, game.reward)
end

# Action effect
function GI.play!(game::GameEnvAudit, action)
	game.circuit = game.circuit(action)
	game.reward = reward(game.circuit,game.adj_m_target)
	return
end

# Termination conditions
GI.game_terminated(game::GameEnvAudit) = game.reward || length(game.circuit.c) ≥ A_MAX_DEPTH

# Vectorize repr of a state, fed to the NN
function GI.vectorize_state(::GameSpecAudit, state)
	c = QCir{Hardware}(state.circuit).m
	t = QCir{Audit}(state.target).m
	c = ANCILLA_ARCH ? c[mask_i,mask_j] : c
	t = ANCILLA_ARCH ? t[mask_i,mask_j] : t
	m = normalize(mapCanonical(adjoint(t)*c))
	vs = Float32[f(m[i,j]) for i in 1:DIM_OUT, j in 1:DIM_OUT, f in [real,imag]]
	return vs
end

# Read state from stdin
function GI.read_state(::GameSpecAudit)
	try
		state = []
		for _ in 1:2
			input = readline()
			input = split(input," ")
			c = map(x -> parse(UInt8,x), input)
			push!(state,c)
		end
		return (target=state[1],circuit=state[2])
	catch _
		return nothing
	end
end

## Additional methods
# Non mandatory methods that can be useful for the game representation

# For minmax player
GI.heuristic_value(::GameEnvAudit) = 0.

function GI.render(game::GameEnvAudit)
	println("TARGET: ")
	print(game.target)
	println()
	println("CIRCUIT: ")
	print(game.circuit)
end

function GI.action_string(gs::GameSpecAudit, a)
	idx = findfirst(==(a), GI.actions(gs))
	return isnothing(idx) ? "?" : H_GATESET[idx].name
end

function GI.parse_action(gs::GameSpecAudit, s)
	idx = findfirst(==(s), [hg.name for hg in H_GATESET])
	return isnothing(idx) ? nothing : GI.actions(gs)[idx]
end
