import LinearAlgebra: I
using SparseArrays
using Yao

mutable struct QCir 
	c::Vector{UInt8}
	m::SparseMatrixCSC
	function QCir(c::Vector{UInt8}; M_GATESET=M_GATESET, DIM=DIM)
		m = SparseMatrixCSC{ComplexF64}(I,DIM,DIM)
		for g in c
			m = m*M_GATESET[g]
		end
		new(c, m)
	end
end

QCir() = QCir(Vector{UInt8}())

function (qc::QCir)(g::UInt8)
	@assert 1 ≤ g ≤ L_GATESET
	push!(qc.c,g)
	qc.m = qc.m*M_GATESET[g]
	return qc
end

function Base.print(qc::QCir)
	for e in qc.c
		g = GATESET_REF[e]
		println("$(g[2]) → $(g[3])")
	end
end

Yao.mat(qc::QCir) = qc.m

function Yao.chain(qc::QCir)
	c = chain(MODE)
	for e in qc.c
		push!(c, GATESET[e])
	end
	return c
end

mapCanonical(qc::QCir) = mapCanonical(qc.m)

function isRedundant(c::Vector{UInt8},gate::UInt8;gates_ref=GATESET_REF)
	gate = gates_ref[gate] 
	mode = gate[3]
	for e in reverse(c)
		g = gates_ref[e]
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

isRedundant(qc::QCir,gate::UInt8) = isRedundant(qc.c,gate)

function randQCir(;L_GATESET=L_GATESET,MAX_TARGET_DEPTH=MAX_TARGET_DEPTH)
	circuitLength = rand(1:MAX_TARGET_DEPTH)
	k = 0; c = Vector{UInt8}();
	while k < circuitLength
		g = UInt8(rand(1:L_GATESET))
		if isRedundant(c,g)
			continue
		else
			push!(c,g)
			k += 1
		end
	end
	return QCir(c)
end
