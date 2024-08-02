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

if !ANCILLA_ARCH
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
else
	function buildGateSet(modes::Int, ancilla_gs::Dict; reverse_cg=A_REVERSE_CTRL, ancilla_modes=ANCILLA_MODE)
		# Specific gateset builder for ancilla architecture
		# single qubit gates only on ancilla qubits
		gateset = []
		ancilla_q = (modes-ancilla_modes+1):modes
		for g in ancilla_gs["single_gate"]
			for m in ancilla_q
				e = Gate(g,m)
				push!(gateset,e)
			end
		end
		# add all controlled gates (only between ancilla and normal modes)
		for g in ancilla_gs["ctrl_gate"]
			for c in 1:modes-ancilla_modes
				for m in ancilla_q
					e = CtrlGate(g,m,[c])
					push!(gateset,e)
					if reverse_cg
						e = CtrlGate(g,c,[m])
						push!(gateset,e)
					end
				end
			end
		end
		if modes == 2
			return gateset
		end
		# same for cc gates 
		for g in ancilla_gs["cctrl_gate"]
			for c1 in 1:modes-ancilla_modes
				for c2 in 1:modes-ancilla_modes
					for m in ancilla_q
						if c1 != c2
							e = CtrlGate(g,m,[c1,c2])
							push!(gateset,e)
						end
					end
				end
			end
		end
		return gateset
	end
end

function buildCommutationDict(gset)
	d = Dict{Int, Vector{Int}}()
	l = length(gset)
	for i in 1:l
		d[i] = Int[]
		for j in 1:l
			c = sparse(gset[i].mat * gset[j].mat - gset[j].mat * gset[i].mat)
            if c.nzval == ComplexF64[]
				push!(d[i],j)
            end
		end
	end
	return d
end

function buildRedudancyDict(gset)
	redundant = Dict{Int, Vector{Vector{Int}}}()
	# Single gate redundancy
	for i in 1:length(gset)
		redundant[i] = Int[]
	end
	for (i,lastGate) in enumerate(gset)
		for (j,newGate) in enumerate(gset)
			if lastGate.mat == adjoint(newGate.mat)
				push!(redundant[j],[i]) 
			end
		end
	end
	# Two gate redundancy insert gate other mode
	#for (l,n) in redundant
	#	for (i, g) in enumerate(gset)
	#		lg = gset[l[1]]
	#		modes = isa(lg,Gate) ? [lg.target] : [lg.target, lg.ctrl]
	#		if g!=l && g!=n
	#			if isa(g,Gate)
	#				if g.target ∉ modes
	#					redundant[[l[1],i]] = n
	#				end
	#			else
	#				if (g.target ∉ modes) && (g.ctrl ∉ modes)
	#					redundant[[l[1],i]] = n
	#				end
	#			end
	#		end
	#	end
	#end
	# No redundant Swaps
	for (i,cx1) in enumerate(gset)
		# Select CNOT
		if isa(cx1,CtrlGate) && cx1.g == X && length(cx1.ctrl) == 1
			for (j,cx2) in enumerate(gset) 
				# Also select CNOT
				if isa(cx2,CtrlGate) && cx2.g == X && length(cx2.ctrl) == 1
					# Select reverse CNOT
					if cx1.target == cx2.ctrl[1] && cx2.target == cx1.ctrl[1]
						push!(redundant[j],[i,j,i,j,i]) 
					end
				end
			end
		end
	end
	return redundant
end

function isRedundant(c::Vector{UInt8}, g::UInt8, red::Dict, com::Dict) :: Bool
	lc = length(c)
	lc == 0 && return false
	rd = red[g]
	for l in lc:-1:1
		c_ =  c[1:l]
		for r in rd
			lr = length(r) # length of redundancy strin
			lr > l && continue # continue if length greater than circuit length
			c_[end-l+1:end] == r && return true # if matches redundancy string -> true
			# println("\t check : ", c_[end-l+1:end], " vs ", r)
		end
		# continue while it commutes
		g ∉ com[c_[end]] && return false
		# println(g, " commute with: ", c_[end])
	end
	return false
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
