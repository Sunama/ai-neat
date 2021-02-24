# frozen_string_literal: true

module Ai
  module Neat
    class Network
      attr_accessor :layers

      def initialize(models)
        @layers = []

        models.each do |model|
          @layers.push(Layer.new(model[:node_count], model[:node_type], model[:activationfunc]))
        end

        (0..(@layers.count - 2)).each do |i|
          @layers[i].connect(@layers[i + 1].nodes.count)
        end
      end

      def feed_forward
        (0..(@layers.count - 2)).each do |i|
          @layers[i].feed_forward(@layers[i + 1])
        end
      end
    end
  end
end
