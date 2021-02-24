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
          @network.layers[i].nodes.each do |node|
            node.weights.each do |_weight|
              weight = genes.first
              genes.shift
            end
          end

          @network.layers[i].bias.weights.each do |_weight|
            weight = genes.first
            genes.shift
          end
        end
      end

      def feed_forward
        @network.feed_forward
      end

      def inputs
        @network.layers.first.values
      end

      def inputs=(values)
        @network.layers.first.values = values
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
    end
  end
end
