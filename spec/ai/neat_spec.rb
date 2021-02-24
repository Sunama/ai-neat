# frozen_string_literal: true

RSpec.describe Ai::Neat do
  it 'has a version number' do
    expect(Ai::Neat::VERSION).not_to be nil
  end

  it 'config' do
    config = {
      models: [
        { node_count: 5, node_type: :input },
        { node_count: 3, node_type: :output, activationfunc: :softmax }
      ],
      mutation_rate: 0.1,
      crossover_method: :random,
      mutation_method: :random,
      population_size: 10
    }

    neat = Ai::Neat::Neat.new(config)

    expect(neat.creatures.count).to eq(10)
    expect(neat.population_size).to eq(10)
    expect(neat.mutation_rate).to eq(0.1)

    expect(neat.creatures.first.network.layers.count).to eq(2)
    expect(neat.creatures.first.network.layers.first.nodes.count).to eq(5)
    expect(neat.creatures.first.network.layers.last.nodes.count).to eq(3)

    expect(neat.creatures.first.network.layers.first.bias).to_not be_nil
    expect(neat.creatures.first.network.layers.last.bias).to be_nil
  end

  it 'test 2 layer (input & output)' do
    population_size = 100

    config = {
      models: [
        { node_count: 5, node_type: :input },
        { node_count: 3, node_type: :output, activationfunc: :softmax }
      ],
      mutation_rate: 0.1,
      crossover_method: :random,
      mutation_method: :random,
      population_size: population_size
    }

    neat = Ai::Neat::Neat.new(config)

    scores = population_size.times.map{ 0 }
    first_score = 0

    #Generation
    100.times.each do |gen|
      scores = population_size.times.map{ 0 }

      #play
      50.times.each do
        inputs = 5.times.map{ rand(-1.0..1.0) }

        (0..(scores.count - 1)).each do |i|
          neat.set_inputs(inputs, i)
        end

        neat.feed_forward
        decisions = neat.decisions

        (0..(scores.count - 1)).each do |i|
          if inputs.last > 0
            case decisions[i]
            when 0
              scores[i] += 1
            when 1
              scores[i] -= 1
            end
          else
            case decisions[i]
            when 0
              scores[i] -= 1
            when 1
              scores[i] += 1
            end
          end

          scores[i] = 0 if scores[i] < 0
        end
      end

      (0..(scores.count - 1)).each do |i|
        neat.set_fitness(scores[i], i)
      end

      expect(neat.best_creature).to be >= 0
      expect(neat.best_creature).to be < population_size

      first_score = scores[neat.best_creature] if gen == 0

      neat.do_gen
    end

    expect(scores[neat.best_creature]).to be > first_score
  end

  it 'test 3 layer (input & middle & output)' do
    population_size = 100

    config = {
      models: [
        { node_count: 5, node_type: :input },
        { node_count: 5, node_type: :middle, activationfunc: :softmax },
        { node_count: 3, node_type: :output, activationfunc: :softmax }
      ],
      mutation_rate: 0.1,
      crossover_method: :random,
      mutation_method: :random,
      population_size: population_size
    }

    neat = Ai::Neat::Neat.new(config)

    scores = population_size.times.map{ 0 }
    first_score = 0

    #Generation
    100.times.each do |gen|
      scores = population_size.times.map{ 0 }

      #play
      50.times.each do
        inputs = 5.times.map{ rand(-1.0..1.0) }

        (0..(scores.count - 1)).each do |i|
          neat.set_inputs(inputs, i)
        end

        neat.feed_forward
        decisions = neat.decisions

        (0..(scores.count - 1)).each do |i|
          if inputs.last > 0
            case decisions[i]
            when 0
              scores[i] += 1
            when 1
              scores[i] -= 1
            end
          else
            case decisions[i]
            when 0
              scores[i] -= 1
            when 1
              scores[i] += 1
            end
          end

          scores[i] = 0 if scores[i] < 0
        end
      end

      (0..(scores.count - 1)).each do |i|
        neat.set_fitness(scores[i], i)
      end

      expect(neat.best_creature).to be >= 0
      expect(neat.best_creature).to be < population_size

      first_score = scores[neat.best_creature] if gen == 0

      neat.do_gen
    end

    expect(scores[neat.best_creature]).to be > first_score
  end
end
