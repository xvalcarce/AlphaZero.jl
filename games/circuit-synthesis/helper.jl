import LinearAlgebra: det, Diagonal, isdiag
import Random: AbstractRNG
using SparseArrays
using Distributions

abstract type AbstractGate end

struct Gate <: AbstractGate
	g::AbstractBlock
	name::String
	target::Int
	mat::Union{SparseMatrixCSC,Diagonal}
	function Gate(g::AbstractBlock, t::Int)
		name = string(g)*"_"*string(t)
		m = sparse(mat(put(t=>g)(MODE)))
		if isdiag(m)
			m = Diagonal(m.nzval)
		end
		new(g,name,t,m)
	end
end

struct CtrlGate <: AbstractGate
	g::AbstractBlock
	name::String
	target::Int
	ctrl::Vector{Int}
	mat::Union{SparseMatrixCSC,Diagonal}
	function CtrlGate(g::AbstractBlock,t::Int,c::Vector{Int})
		name = "($(string(c)))_$(string(g))_$(string(t))"
		m = sparse(mat(control(c, t=>g)(MODE)))
		if isdiag(m)
			m = Diagonal(m.nzval)
		end
		new(g,name,t,c,m)
	end
end

prettyprint(g::Gate) = println("$(g.g) → $(g.target)")
prettyprint(g::CtrlGate) = println("$(g.g) → $(g.target) ◌ $(g.ctrl)")

function buildGateSet(modes::Int, gs::Dict)
	# gate constructor for gates on ALL qubits
	gateset = []
	for g in gs["single_gate"]
		for t in 1:modes
			e = Gate(g,t)
			push!(gateset,e)
		end
	end
	if modes == 1
		return gateset
	end
	# add all controlled gates
	for g in gs["ctrl_gate"]
		for t in 1:modes
			for c in 1:modes
				if t != c
					e = CtrlGate(g,t,[c])
					push!(gateset,e)
				end
			end
		end
	end
	if modes == 2
		return gateset
	end
	for g in gs["cctrl_gate"]
		for t in 1:modes
			for c1 in 1:modes
				for c2 in 1:modes
					if t != c1 && t != c2 && c1 != c2
						e = CtrlGate(g,t,[c1,c2])
						push!(gateset,e)
					end
				end
			end
		end
	end
	return gateset
end

function buildRedudancyDict(gset)
	redundant = Dict{Vector{Int}, Int}()
	# Single gate redundancy
	for (i,lastGate) in enumerate(gset)
		for (j,newGate) in enumerate(gset)
			if lastGate.mat == adjoint(newGate.mat)
				redundant[[i]] = j
			end
		end
	end
	# Two gate redundancy insert gate other mode
	for (l,n) in redundant
		for (i, g) in enumerate(gset)
			lg = gset[l[1]]
			modes = isa(lg,Gate) ? [lg.target] : [lg.target, lg.ctrl]
			if g!=l && g!=n
				if isa(g,Gate)
					if g.target ∉ modes
						redundant[[l[1],i]] = n
					end
				else
					if (g.target ∉ modes) && (g.ctrl ∉ modes)
						redundant[[l[1],i]] = n
					end
				end
			end
		end
	end
	# Swap gate
	for (i,cx1) in enumerate(gset)
		# Select CNOT
		if isa(cx1,CtrlGate) && cx1.g == X && length(cx1.ctrl) == 1
			for (j,cx2) in enumerate(gset) 
				# Also select CNOT
				if isa(cx2,CtrlGate) && cx2.g == X && length(cx2.ctrl) == 1
					# Select reverse CNOT
					if cx1.target == cx2.ctrl[1] && cx2.target == cx1.ctrl[1]
						redundant[[i,j,i]] = j 
					end
				end
			end
		end
	end
	return redundant
end

if USE_GP_SYM
	function mapCanonical(u::SparseMatrixCSC)
		N = u.n
		su_mat = u/(det(u)^(1/N)) #Convert Matrix to su(n)
		nz = round(su_mat.nzval[1], digits=12) #take first nonzero
		hs = [hash(round(exp(-im*2*π*i/N)*nz,digits=12)) for i in 1:N] #hacky af but works : hash all 8 possible repr of nz
		su_uniq = round.(exp(-im*2*π*argmin(hs)/N)*su_mat, digits=12) #round is super helpful, helps for hashing
		return su_uniq
	end

	function mapCanonical(u::Diagonal)
		N = size(u)[1]
		su_mat = u/(det(u)^(1/N)) #Convert Matrix to su(n)
		nz = round(u.diag[findfirst(x-> x!=0.0,u.diag)], digits=12) #take first nonzero
		hs = [hash(round(exp(-im*2*π*i/N)*nz,digits=12)) for i in 1:N] #hacky af but works : hash all 8 possible repr of nz
		su_uniq = round.(exp(-im*2*π*argmin(hs)/N)*su_mat, digits=14) #round is super helpful, helps for hashing
		return su_uniq
	end
else
	mapCanonical(u::Union{SparseMatrixCSC, Diagonal}) = round.(u, digits=14)
end

struct BiasUniform <: Sampleable{Univariate,Discrete} end

function Base.rand(s::BiasUniform)
	range = rand()
	if range > WEIGHT
		rand(HALF_TARGET_DEPTH:MAX_TARGET_DEPTH)
	else
		rand(MIN_TARGET_DEPTH:HALF_TARGET_DEPTH)
	end
end
