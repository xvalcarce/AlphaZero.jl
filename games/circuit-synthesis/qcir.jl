import LinearAlgebra: I
using SparseArrays
using Yao

abstract type Architecture end

struct Target <: Architecture end
struct Hardware <: Architecture end

# Coresponding matrices 
gateset(::Type{Target}) = T_GATESET
gateset(::Type{Hardware}) = H_GATESET

mutable struct QCir{T<:Architecture}
	c::Vector{UInt8}
	m::SparseMatrixCSC
	function QCir{T}(c::Vector{UInt8}; DIM=DIM) where T <: Architecture
		m = MAT_ID
		gset = gateset(T)
		for g in c
			m = m*gset[g].mat
		end
		new(c, m)
	end
end

QCir{T}() where T <: Architecture = QCir{T}(Vector{UInt8}())

function (qc::QCir{T})(g::UInt8) where T <: Architecture
	gset = gateset(T)
	@assert 1 ≤ g ≤ length(gset)
	push!(qc.c,g)
	qc.m = qc.m*gset[g].mat
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

function isRedundant(c::Vector{UInt8},gate::UInt8,gset::Vector{Any})::Bool
	if length(c) == 0
		return false
	elseif gate == c[end] #Only check if same gate
		ishermitian(gset[gate].g) && return true # If hermitian then redundant
	else	
		return false
	end
end

isRedundant(qc::QCir{T},gate::UInt8) where T<:Architecture = isRedundant(qc.c,gate,gateset(T))

function Base.rand(::Type{T},circuitLength::Int) where T<:Architecture
	gset = gateset(T)
	l = length(gset);
	k = 0; c = Vector{UInt8}();
	while k < circuitLength
		g = UInt8(rand(1:l))
		if isRedundant(c,g,gset)
			continue
		else
			push!(c,g)
			k += 1
		end
	end
	return QCir{T}(c)
end

Base.rand(::Type{T}) where T<:Architecture = rand(T,rand(1:MAX_TARGET_DEPTH))

function weightedRand(::Type{T}) where T<:Architecture
	range = rand()
	if range > WEIGHT
		return rand(T,rand(HALF_TARGET_DEPTH:MAX_TARGET_DEPTH))
	else
		return rand(T,rand(1:HALF_TARGET_DEPTH))
	end
end
