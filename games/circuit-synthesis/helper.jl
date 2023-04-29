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
					push!(gates_name,"($(string(c)))_$(string(g))_$(string(t))")
				end
			end
		end
	end
	return lowercase.(gates_name)
end

#TODO: redudancy check -> return mask for red/non-red gates
function isRedundant(c::ChainBlock,gate)
	gate = gate(c.n) # Slow...
	if isa(gate,ControlBlock)
		mode = [gate.ctrl_locs[1], gate.locs[1]]
	else
		mode = [gate.locs[1]]
	end	
	for g in reverse(c.blocks)
		if isa(g,ControlBlock)
			if any(mode .∈ Ref([g.ctrl_locs[1], g.locs[1]]))
				r = g == gate ? true : false
				return r
			end
		else
			if g.locs[1] ∈ mode 
				r = g == gate' ? true : false
				return r
			end
		end
	end
	return false
end

function randomCircuit(MODE::Int,GATESET::Vector,max_depth=MAX_TARGET_DEPTH)
	""" Generate a random circuit """
	l = length(GATESET)
	u = chain(MODE)
	circuitLength = rand(1:max_depth)
	k = 0
	while k < circuitLength
		r = rand(1:l)
		if isRedundant(u,GATESET[r])
			continue
		else
			push!(u,GATESET[r])
			k += 1
		end
	end
	return u
end
