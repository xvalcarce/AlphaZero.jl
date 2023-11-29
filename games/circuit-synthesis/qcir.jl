import LinearAlgebra: I
using SparseArrays
using Yao

abstract type Architecture end

struct Target <: Architecture end
struct Hardware <: Architecture end

# Coresponding matrices 
gateset_m(::Type{Target}) = T_GATESET_M
gateset_m(::Type{Hardware}) = H_GATESET_M

# Coresponding gate set references
gateset_r(::Type{Target}) = T_GATESET_REF
gateset_r(::Type{Hardware}) = H_GATESET_REF

# Coresponding length of gateset 
gateset_l(::Type{Target}) = T_GATESET_L
gateset_l(::Type{Hardware}) = H_GATESET_L

mutable struct QCir{T<:Architecture}
	c::Vector{UInt8}
	m::SparseMatrixCSC
	function QCir{T}(c::Vector{UInt8}; DIM=DIM) where T <: Architecture
		m = SparseMatrixCSC{ComplexF64}(I,DIM,DIM)
		matrices = gateset_m(T)
		for g in c
			m = m*matrices[g]
		end
		new(c, m)
	end
end

QCir{T}() where T <: Architecture = QCir{T}(Vector{UInt8}())

function (qc::QCir{T})(g::UInt8) where T <: Architecture
	matrices = gateset_m(T)
	@assert 1 ≤ g ≤ length(matrices)
	push!(qc.c,g)
	qc.m = qc.m*matrices[g]
	return qc
end

function Base.print(qc::QCir{T}) where T <: Architecture
	g_ref = gateset_r(T) 
	for e in qc.c
		g = g_ref[e]
		println("$(g[2]) → $(g[3])")
	end
end

function Base.copy(qc::QCir{T}) where T <: Architecture
	c = copy(qc.c)
	return QCir{T}(c)
end

mapCanonical(qc::QCir) = mapCanonical(qc.m)

function isRedundant(c::Vector{UInt8},gate::UInt8,gateset_ref::Vector{Any})::Bool
	gate = gateset_ref[gate] 
	mode = gate[3]
	for e in reverse(c)
		g = gateset_ref[e]
		if g[1]
			if any(mode .∈ Ref(g[3]))
				r = g == gate ? true : false
				return r
			end
		else
			if g[3][1] ∈ mode 
				r = g[2] == gate[2]' ? true : false
				return r
			end
		end
	end
	return false
end

isRedundant(qc::QCir{T},gate::UInt8) where T<:Architecture = isRedundant(qc.c,gate,gateset_r(T))

function Base.rand(::Type{T},circuitLength::Int) where T<:Architecture
	l = gateset_l(T); g_ref = gateset_r(T);
	k = 0; c = Vector{UInt8}();
	while k < circuitLength
		g = UInt8(rand(1:l))
		if isRedundant(c,g,g_ref)
			continue
		else
			push!(c,g)
			k += 1
		end
	end
	return QCir{T}(c)
end

Base.rand(::Type{T}) where T<:Architecture = rand(T,rand(1:MAX_TARGET_DEPTH))
