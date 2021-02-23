module Ai
  module Neat
    class Layer
      attr_accessor :nodes, :bias, :node_type, :activationfunc

      def initialize(node_count, node_type, activationfunc)
        @nodes = []
        @bias = nil
        @node_type = node_type
        @activationfunc = activationfunc

        (1..node_count).each do |i|
          @nodes.push(Node.init())
        end

        @bias = Node.init() if @node_type != :output
      end

      def connect(count)
        @nodes.each do |node|
          node.initWeights(count)
        end

        @bias.initWeights(count) if @bias
      end

      def feed_forward(layer)
        (0..(@bias.weights.count - 1)).each do |i|
          layer.nodes[i].value = 0
        end

        @nodes.each do |node|
          (0..(node.weights.count - 1)).each do |w|
            layer.nodes[w].value = node.value * node.weights[w]
          end
        end

        (0..(@bias.weights.count - 1)).each do |w|
          layer.nodes[w].value += @bias.weights[w]
        end

        if layer.activationfunc != :softmax
          layer.nodes.each do |node|
            node.value = Ai::Neat::activationfunc(layer.activationfunc, node.value)
          end
        else
          layer.values(Ai::Neat::activationfunc(layer.activationfunc, layer.values()))
        end
      end

      def values
        result = []

        @nodes.each do |node|
          result.push(node.value)
        end

        result
      end

      def values=(values)
        (0..(@nodes.count - 1)).each do |i|
          @nodes[i].value = values[i]
        end
      end
    end
  end
end