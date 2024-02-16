module CircuitSynthesis
  export GameSpec, GameEnv
  using AlphaZero
  include("game.jl")
  include("test.jl")
  module Training
    using AlphaZero
    import ..GameSpec
    include("params.jl")
  end
end
