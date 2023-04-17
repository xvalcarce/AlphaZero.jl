
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

function gateset_name(modes::Int, gateset::Vector, ctrl_set::Vector)
	# gate constructor for gates on ALL qubits
	gates_name = []
	for g in gateset
		for bit in 1:modes
			push!(gates_name,string(g)*"_"*string(bit))
		end
	end
	# add all controlled gates
	for g in ctrl_set
		for t in 1:modes
			for c in 1:modes
				if t != c
					push!(gates_name,"($(string(c)))_"*string(g)*"_"*string(t))
				end
			end
		end
	end
	return gates_name
end

function randomCircuit(MODE::Int,GATESET::Vector,depth=3)
	""" Generate a random circuit """
	l = length(GATESET)
	u = chain(MODE)
	for _ in 1:depth
		r = rand(1:l)
		push!(u,GATESET[r])
	end
	return u
end
