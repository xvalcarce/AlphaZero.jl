using ProgressMeter
using Statistics: mean, std

"""
	Custom benchmark function for Circuit-Synthesis game spec.
	
"""
function Benchmark.run(env::Env{<:GameSpec,<:Any,<:Any}, eval::Benchmark.Evaluation, progress=nothing)
  net() = Network.copy(env.bestnn, on_gpu=eval.sim.use_gpu, test_mode=true)
  gspec = GameSpecAudit()
  if isa(eval, Benchmark.Single)
    simulator = Simulator(net, record_trace) do net
      Benchmark.instantiate(eval.player, gspec, net)
    end
  else
    @assert isa(eval, Benchmark.Duel)
    simulator = Simulator(net, record_trace) do net
      player = Benchmark.instantiate(eval.player, gspec, net)
      baseline = Benchmark.instantiate(eval.baseline, gspec, net)
      return TwoPlayers(player, baseline)
    end
  end
  samples, elapsed = @timed simulate(
    simulator, gspec, eval.sim,
    game_simulated=(() -> next!(progress)))
  gamma = env.params.self_play.mcts.gamma
  rewards, redundancy = rewards_and_redundancy(samples, gamma=gamma)
  return Report.Evaluation(
    Benchmark.name(eval), mean(rewards), redundancy, rewards, nothing, elapsed)
end
