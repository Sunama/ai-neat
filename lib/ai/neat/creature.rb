# frozen_string_literal: true

module Ai
  module Neat
    class Creature
      attr_accessor :network, :fitness, :score

      def initialize(models)
        @network = Network.new(models)
        @fitness = 0
        @score = 0
      end

      def decision
        index = -1
        max = -Float::INFINITY

        (0..(@network.layers.last.nodes.count - 1)).each do |i|
          if @network.layers.last.nodes[i].value > max
            max = @network.layers.last.nodes[i].value
            index = i
          end
        end

        index
      end

      def feed_forward
        @network.feed_forward
      end

      def flatten_genes
        genes = []

        (0..(@network.layers.count - 2)).each do |i|
          @network.layers[i].nodes.each do |node|
            node.weights.each do |weight|
              genes.push(weight)
            end
          end

          @network.layers[i].bias.weights.each do |weight|
            genes.push(weight)
          end
        end

        genes
      end

      def flatten_genes=(genes)
        (0..(@network.layers.count - 2)).each do |i|
          (0..@network.layers[i].nodes.count - 1).each do |w|
            (0..@network.layers[i].nodes[w].weights.count - 1).each do |e|
              @network.layers[i].nodes[w].weights[e] = genes.first
              genes.shift
            end
          end
    
          (0..@network.layers[i].bias.weights.count - 1).each do |w|
            @network.layers[i].bias.weights[w] = genes.first
            genes.shift
          end
        end
      end

      def inputs
        @network.layers.first.values
      end

      def inputs=(values)
        @network.layers.first.values = values
      end
    end
  end
end
