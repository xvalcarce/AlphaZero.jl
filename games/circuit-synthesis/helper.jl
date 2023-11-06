import LinearAlgebra: det
using SparseArrays

function gateset(modes::Int, gateset::Vector, ctrl_set::Vector)
	# gate constructor for gates on ALL qubits
	gates = []
	gates_name = []
	gates_ref = []
	for g in gateset
		for bit in 1:modes
			push!(gates, put(bit=>g))
			push!(gates_name,string(g)*"_"*string(bit))
			push!(gates_ref,[false,g,[bit]])
		end
	end
	# add all controlled gates
	for g in ctrl_set
		for t in 1:modes
			for c in 1:modes
				if t != c
					push!(gates, control(c, t=>g))
					push!(gates_name,"($(string(c)))_$(string(g))_$(string(t))")
					push!(gates_ref,[true,g,[c,t]])
				end
			end
		end
	end
	return gates, lowercase.(gates_name), gates_ref
end

function mapCanonical(u::SparseMatrixCSC)
	N = u.n
	su_mat = u/(det(u)^(1/N)) #Convert Matrix to su(n)
	nz = su_mat.nzval[1] #take first nonzero
	hs = [hash(round(exp(-im*2*π*i/N)*nz,digits=12)) for i in 1:N] #hacky af but works : hash all 8 possible repr of nz
	su_uniq = round.(exp(-im*2*π*argmin(hs)/N)*su_mat, digits=14) #round is super helpful, helps for hashing
	return su_uniq
end

function mapCanonical(u::Diagonal)
	N = size(u)[1]
	su_mat = u/(det(u)^(1/N)) #Convert Matrix to su(n)
	nz = u.diag[findfirst(x-> x!=0.0,u.diag)] #take first nonzero
	hs = [hash(round(exp(-im*2*π*i/N)*nz,digits=12)) for i in 1:N] #hacky af but works : hash all 8 possible repr of nz
	su_uniq = round.(exp(-im*2*π*argmin(hs)/N)*su_mat, digits=14) #round is super helpful, helps for hashing
	return su_uniq
end
