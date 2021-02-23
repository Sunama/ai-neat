# frozen_string_literal: true

require_relative "neat/version"

module Ai
  module Neat
    class Neat
      attr_accessor :creatures, :old_creatures, :models, :export_model, :population_size, :mutation_rate,
        :crossover_method, :mutation_method, :generation
      
      def initialize(config)
        @creatures = []
        @old_creatures = []
        @models = config.models
        @export_model = []
        @population_size = config.population_size || 500
        @mutation_rate = config.mutation_rate || 0.05
        @crossover_method = config.crossover_method || :random
        @mutation_method = config.mutation_method || :random
        @generation = 0

        @models.each do |model|
          @export_model.push(model.clone)
        end

        (1..@population_size).each do |i|
          @creatures.push(Creature.init(@models))
        end
      end

      def mutate
        @creatures.each do |creature|
          genes = creature.flatten_genes
          genes = mutation(@mutation_method, @mutation_rate)
          creature.flatten_genes = genes
        end
      end

      def crossover
        @creatures.each do |creature|
          @old_creatures = @creatures.clone
          parent_x = pick_creature
          parent_y = pick_creature

          genes = Ai::Neat::crossover(parent_x.flatten_genes, parent_y.flatten_genes)
          creature.flatten_genes = genes
        end
      end

      def pick_creature
        sum = 0

        @old_creatures.each do |creature|
          sum += creature.score ^ 2
        end

        @old_creatures.each do |creature|
          creature.fitness = (creature.score ^ 2) / sum
        end

        index = 0
        r = rand()
        while r > 0
          r -= @old_creatures[index].fitness
          index += 1
        end

        index -= 1

        @old_creatures[index]
      end

      def set_fitness(fitness, index)
        @creatures[index].score = fitness
      end

      def feed_forward
        @creatures.each do |creature|
          creature.feed_forward
        end
      end

      def do_gen
        crossover()
        mutate()
        @generation += 1
      end

      def best_creature
        index = 0
        max = -Infinity

        (0..(@old_creatures.count - 1)).each do |i|
          if @old_creatures[i].fitness > max
            max = @old_creatures[i].fitness
            index = i
        end

        index
      end

      def decisions
        result = []

        @creatures.each do |creature|
          result.push(creature.decision())
        end

        result
      end

      def set_inputs(inputs, index)
        @creatures[index].inputs = inputs
      end

      def export(index = nil)
        
      end

      def import(data)

      end
    end
  end
end
