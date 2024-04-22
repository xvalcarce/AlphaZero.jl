using ProgressMeter
using Statistics: mean, std

@kwdef struct SingleAudit <: Benchmark.Evaluation
  player :: Benchmark.Player
  sim :: SimParams
end

Benchmark.name(s::SingleAudit) = Benchmark.name(s.player)*" Audit"

"""
	Custom benchmark function for Circuit-Synthesis game spec.
	
"""
function Benchmark.run(env::Env{<:GameSpec,<:Any,<:Any}, eval::Benchmark.Evaluation, progress=nothing)
	net() = Network.copy(env.bestnn, on_gpu=eval.sim.use_gpu, test_mode=true)
	if isa(eval,SingleAudit)
		gspec = GameSpecAudit()
	else
		gspec = GameSpec()
	end
	simulator = Simulator(net, record_trace) do net
		Benchmark.instantiate(eval.player, gspec, net)
	end
	samples, elapsed = @timed simulate(
		simulator, gspec, eval.sim,
    	game_simulated=(() -> next!(progress)))
  	gamma = env.params.self_play.mcts.gamma
  	rewards, redundancy = rewards_and_redundancy(samples, gamma=gamma)
  	return Report.Evaluation(
    	Benchmark.name(eval), mean(rewards), redundancy, rewards, nothing, elapsed)
end
