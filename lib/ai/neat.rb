# frozen_string_literal: true

Dir[File.dirname(__FILE__) + "/neat/*.rb"].sort.each { |file| require file }

module Ai
  module Neat
    class Neat
      attr_accessor :creatures, :old_creatures, :models, :population_size, :mutation_rate,
                    :crossover_method, :mutation_method, :generation

      def initialize(config)
        @creatures = []
        @old_creatures = []
        @models = config[:models]
        @population_size = config[:population_size] || 500
        @mutation_rate = config[:mutation_rate] || 0.05
        @crossover_method = config[:crossover_method] || :random
        @mutation_method = config[:mutation_method] || :random
        @generation = 0

        (1..@population_size).each do |_i|
          @creatures.push(Creature.new(@models))
        end
      end

      def best_creature
        index = 0
        max = -Float::INFINITY

        (0..(@old_creatures.count - 1)).each do |i|
          if @old_creatures[i].fitness > max
            max = @old_creatures[i].fitness
            index = i
          end
        end

        index
      end

      def crossover
        (0..@population_size - 1).each do |i|
          parent_x = pick_creature
          parent_y = pick_creature

          genes = Ai::Neat.crossover(@crossover_method, parent_x.flatten_genes, parent_y.flatten_genes)
          @creatures[i].flatten_genes = genes
        end
      end

      def decisions
        result = []

        @creatures.each do |creature|
          result.push(creature.decision)
        end

        result
      end

      def do_gen
        @old_creatures = @creatures.clone
        
        crossover
        mutate
        @generation += 1
      end

      def export
        data = {
          models: @models,
          creatures: []
        }

        @creatures.each do |creature|
          data[:creatures].push(creature.flatten_genes)
        end

        data
      end

      def feed_forward
        @creatures.each(&:feed_forward)
      end

      def import(data)
        @models = data[:models]

        @creatures = []
        @population_size = 0

        data[:creatures].each do |genes|
          creature = Creature.new(@models)
          creature.flatten_genes = genes

          @creatures.push(creature)

          @population_size += 1
        end
      end

      def mutate
        (0..@population_size - 1).each do |i|
          genes = @creatures[i].flatten_genes
          genes = Ai::Neat.mutate(@mutation_method, genes, @mutation_rate)
          @creatures[i].flatten_genes = genes
        end
      end

      def pick_creature
        sum = 0.0

        @old_creatures.each do |creature|
          sum += creature.score ** 2
        end

        @old_creatures.each do |creature|
          creature.fitness = (creature.score ** 2) / sum
        end

        index = 0
        r = rand

        while r.positive?
          r -= @old_creatures[index].fitness
          index += 1
        end

        index -= 1

        @old_creatures[index]
      end

      def set_fitness(fitness, index)
        @creatures[index].score = fitness
      end

      def set_inputs(inputs, index)
        @creatures[index].inputs = inputs
      end
    end
  end
end
