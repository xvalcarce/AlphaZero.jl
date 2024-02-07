import LinearAlgebra: det
using SparseArrays

abstract type AbstractGate end

struct Gate <: AbstractGate
	g::AbstractBlock
	name::String
	target::Int
	mat::SparseMatrixCSC
	function Gate(g::AbstractBlock, t::Int)
		name = string(g)*"_"*string(t)
		m = sparse(mat(put(t=>g)(MODE)))
		new(g,name,t,m)
	end
end

struct CtrlGate <: AbstractGate
	g::AbstractBlock
	name::String
	target::Int
	ctrl::Vector{Int}
	mat::SparseMatrixCSC
	function CtrlGate(g::AbstractBlock,t::Int,c::Vector{Int})
		name = "($(string(c)))_$(string(g))_$(string(t))"
		m = sparse(mat(control(c, t=>g)(MODE)))
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
