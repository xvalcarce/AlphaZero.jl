import LinearAlgebra: I, normalize
using Distributions
using SparseArrays
using Yao

abstract type Architecture end

struct Target <: Architecture end
struct Hardware <: Architecture end

# Coresponding matrices 
gateset(::Type{Target}) = T_GATESET
gateset(::Type{Hardware}) = H_GATESET

# Redundancy dict
redundancy(::Type{Target}) = T_REDUNDANCY
redundancy(::Type{Hardware}) = H_REDUNDANCY

mutable struct QCir{T<:Architecture}
	c::Vector{UInt8}
	m::SparseMatrixCSC
	function QCir{T}(c::Vector{UInt8}; DIM=DIM) where T <: Architecture
		m = MAT_ID
		gset = gateset(T)
		for g in c
			m = gset[g].mat*m
		end
		new(c, m)
	end
end

QCir{T}() where T <: Architecture = QCir{T}(Vector{UInt8}())

function (qc::QCir{T})(g::UInt8) where T <: Architecture
	gset = gateset(T)
	@assert 1 ≤ g ≤ length(gset)
	push!(qc.c,g)
	qc.m = gset[g].mat*qc.m
	return qc
end

function Base.print(qc::QCir{T}) where T <: Architecture
	gset = gateset(T) 
	for e in qc.c
		prettyprint(gset[e])
	end
end

function Base.copy(qc::QCir{T}) where T <: Architecture
	c = copy(qc.c)
	return QCir{T}(c)
end

mapCanonical(qc::QCir) = mapCanonical(qc.m)

function isRedundant(c::Vector{UInt8}, g::UInt8, red::Dict) :: Bool
	lc = length(c)
	lc == 0 && return false
	for i in 1:min(3,lc)
		if haskey(red, c[end-i+1:end]) 
			red[c[end-i+1:end]] == g && return true
		end
	end
	return false
end

isRedundant(qc::QCir{T},gate::UInt8) where T<:Architecture = isRedundant(qc.c,gate,redundancy(T))

function Base.rand(::Type{T},circuitLength::Int) where T<:Architecture
	gset = gateset(T)
	red = redundancy(T)
	l = length(gset);
	k = 0; c = Vector{UInt8}();
	while k < circuitLength
		g = UInt8(rand(1:l))
		if isRedundant(c,g,red)
			continue
		else
			push!(c,g)
			k += 1
		end
	end
	cir = QCir{T}(c)
	if ANCILLA_ARCH
		# Has to contain at least a CNOT for ANCILLA_ARCH
		if !any(c .∈ H_CTRL_REF)
			cir = rand(T,circuitLength)
		end
		# Select only if a valid unitary operation is produced
		m = cir.m[mask_i,mask_j]
		if !isapprox(normalize(adjoint(m)*m),MAT_ID_OUT)
			cir = rand(T,circuitLength)
		end
	end
	return cir
end

Base.rand(::Type{T}) where T<:Architecture = rand(T,rand(MIN_TARGET_DEPTH:MAX_TARGET_DEPTH))

Distributions.rand(d::Sampleable,::Type{T}) where T<:Architecture = rand(T,Int(round(rand(d))))

function toyao(c::QCir{T}) where T<:Architecture
	gset = gateset(T)
	yc = chain(MODE)
	for g in c.c
		gate = gset[g]
		if isa(gate, Gate)
			push!(yc,put(gate.target=>gate.g))
		elseif isa(gate, CtrlGate)
			push!(yc,control(gate.ctrl, gate.target=>gate.g))
		else
			error("Type of gate unkown")
		end
	end
	return yc
end

toyao(c::Vector{UInt8}) = toyao(QCir{Hardware}(c))

function latexify(c::ChainBlock)
	header = "\\Qcircuit @C=1em @R=.7em  {"
	q = ["" for i in 1:c.n]
	tail = "}"
	newcolumn() = map(x -> x*" & ",q)  
	idc = []
	for g in c
		if isa(g, PutBlock)
			s = string(g.content)
			if length(s) !== 1
				s = s[1]*"^\\dag"
			end
			idx = g.locs[1]
			if idx ∈ idc
				for i in 1:c.n
					if i ∉ idc
						q[i] *= "\\qw"
					end
				end
				q = newcolumn()
				idc = []
			end
			q[idx] = q[idx]*"\\gate{"*s*"}"
			append!(idc,idx)
		elseif isa(g, ControlBlock)
			s = string(g.content)
			if length(s) !== 1
				s = "\\gate{"*s[1]*"^\\dag}"
			elseif s == "X"
				s = "\\targ"
			else
				s = "\\gate{"*s[1]*"}"
			end
			idx = g.locs[1]
			cidx = g.ctrl_locs[1]
			if any(map(x -> x ∈ idc, [idx,cidx]))
				for i in 1:c.n
					if i ∉ idc
						q[i] *= "\\qw"
					end
				end
				q = newcolumn()
				idc = []
			end
			q[idx] *= s
			q[cidx] *= "\\ctrl{$(idx-cidx)}"
			append!(idc,idx)
			append!(idc,cidx)
		end
	end
	for i in 1:c.n
		if i ∉ idc
			q[i] *= "\\qw"
		end
	end
	q = newcolumn()
	q  = map(x -> x*"\\qw", q)
	q  = map(x -> "\\qw & "*x, q)
	for i in 1:c.n-1
		q[i] *= " \\\\"
	end
	return join([header,q...,tail],"\n")
end
