module Ai
  module Neat
    class Creature
      attr_accessor :network, :fitness, :score

      def initialize(models)
        @network = Network.init(models)
        @fitness = 0
        @score = 0
      end

      def flatten_genes
        genes = []

        @network.layers.each do |layer|
          layer.nodes.each do |node|
            node.weights.each do |weight|
              genes.push(weight)
            end
          end

          layer.bias.weights.each do |weight|
            genes.push(weight)
          end
        end

        genes
      end

      def flatten_genes=(genes)
        (0..(@network.layers.count - 2)).each do |i|
          @network.layers[i].nodes.each do |node|
            node.weights.each do |weight|
              weight = genes.first
              genes.shift!
            end
          end

          @network.layers[i].bias.weights.each do |weight|
            weight = genes.first
            genes.shift!
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
        max = -Infinity

        (0..(@network.layers.last.nodes.count)).each do |i|
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