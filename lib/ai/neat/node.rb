# frozen_string_literal: true

module Ai
  module Neat
    class Node
      attr_accessor :value, :weights

      def initialize
        @value = 0
        @weights = []
      end

      def initWeights(count)
        (1..count).each do |_i|
          weights.push(rand(-1.0..1.0))
        end
      end
    end
  end
end
