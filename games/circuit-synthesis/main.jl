module CircuitSynthesis
  export GameSpec, GameEnv
  using AlphaZero
  include("game.jl")
  include("test.jl")
  include("benchmark.jl")
  module Training
    using AlphaZero
    import ..GameSpec, ..SingleAudit
    include("params.jl")
  end
end
