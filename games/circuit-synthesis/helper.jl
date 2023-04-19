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
			if isa(g,ShiftGate)
				gs = g.theta == π/2 ? "S" : "Sdag"
			else
				gs = string(g)
			end
			push!(gates_name,gs*"_"*string(bit))
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
	return lowercase.(gates_name)
end

function randomCircuit(MODE::Int,GATESET::Vector,max_depth=MAX_TARGET_DEPTH)
	""" Generate a random circuit """
	l = length(GATESET)
	u = chain(MODE)
	circuitLength = rand(1:max_depth)
	for _ in 1:circuitLength
		r = rand(1:l)
		push!(u,GATESET[r])
	end
	return u
end
