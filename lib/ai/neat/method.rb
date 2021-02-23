module Ai
  module Neat
    def self.activationfunc(method, value)
      case method
      when :relu
        value > 0 ? value : 0
      when :tanh
        Math.tanh(value)
      when :sigmoid
        (1 / (1 + Math.exp(-value)))
      when :leaky_relu
        value > 0 ? value : (value * 0.01)
      when :softmax
        sum = 0
        result = []

        value.each do |v|
          sum += Math.exp(v)
        end

        value.each do |v|
          result.push(Math.exp(v) / sum)
        end

        result
      end
    end

    def self.crossover(method, genes_x, genes_y)
      case method
      when :random
        genes = []

        (0..(genes_x.count - 1)).each do |i|
          rand() < 0.5 ? genes.push(genes_x[i]) : genes.push(genes_y[i])
        end

        genes
      when :slice
        start_index = (rand() * genes_x.count).to_i
        end_index = (rand() * (genes_x.count - start_index + 2)).to_i + start_index + 1
        cut_genes = genes_x[start_index..end_index]

        genes_y.slice!(start_index..end_index)
        genes_y.concat(cut_genes)

        genes_y
      end
    end

    def self.mutate(method, genes, rate)
      case method
      when :random
        genes.each do |gene|
          gene = rand(-1.0..1.0) if rand() < rate
        end
      when :edit
        genes.each do |gene|
          gene += rand(0.0..0.5) if rand() < rate
        end
      end

      genes
    end
  end
end