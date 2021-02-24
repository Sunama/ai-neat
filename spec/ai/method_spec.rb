RSpec.describe Ai::Neat do
  describe 'activationfunc' do
    it 'relu' do
      expect(Ai::Neat.activationfunc(:relu, 0)).to eq(0)
      expect(Ai::Neat.activationfunc(:relu, -0.1)).to eq(0)
      expect(Ai::Neat.activationfunc(:relu, 0.1)).to eq(0.1)
    end

    it 'tanh' do
      expect(Ai::Neat.activationfunc(:tanh, 0)).to eq(Math.tanh(0))
      expect(Ai::Neat.activationfunc(:tanh, 0.5)).to eq(Math.tanh(0.5))
      expect(Ai::Neat.activationfunc(:tanh, 1)).to eq(Math.tanh(1))
    end

    it 'sigmoid' do
      expect(Ai::Neat.activationfunc(:sigmoid, 0)).to eq((1 / (1 + Math.exp(0))))
      expect(Ai::Neat.activationfunc(:sigmoid, 0.5)).to eq((1 / (1 + Math.exp(-0.5))))
      expect(Ai::Neat.activationfunc(:sigmoid, 1)).to eq((1 / (1 + Math.exp(-1))))
    end

    it 'leaky_relu' do
      expect(Ai::Neat.activationfunc(:leaky_relu, -1)).to eq((-0.01))
      expect(Ai::Neat.activationfunc(:leaky_relu, 0)).to eq(0)
      expect(Ai::Neat.activationfunc(:leaky_relu, 1)).to eq(1)
      expect(Ai::Neat.activationfunc(:leaky_relu, 1.2)).to eq(1.2)
    end

    it 'softmax' do
      values = 100.times.map{ rand(0..10) }
      values = Ai::Neat.activationfunc(:softmax, values)
      expect(values.class.to_s).to eq('Array')
      expect(values.count).to eq(100)
      
      values.each do |value|
        expect(value).to be >= 0
        expect(value).to be <= 1
      end
    end
  end

  describe 'crossover' do
    it 'random' do
      genes_x = 100.times.map{ rand(-1.0..1.0) }
      genes_y = 100.times.map{ rand(-1.0..1.0) }

      genes = Ai::Neat.crossover(:random, genes_x, genes_y)

      expect(genes.count).to eq(100)
      
      changed_x = false
      changed_y = false

      (0..(genes.count - 1)).each do |i|
        expect(genes[i]).to be >= -1
        expect(genes[i]).to be <= 1

        changed_x = true if genes[i] != genes_x[i]
        changed_y = true if genes[i] != genes_y[i]
      end

      expect(changed_x).to be(true)
      expect(changed_y).to be(true)
    end

    it 'slice' do
      genes_x = 100.times.map{ rand(-1.0..1.0) }
      genes_y = 100.times.map{ rand(-1.0..1.0) }

      genes = Ai::Neat.crossover(:slice, genes_x, genes_y)

      expect(genes.count).to eq(100)

      changed_x = false
      changed_y = false

      (0..(genes.count - 1)).each do |i|
        expect(genes[i]).to be >= -1
        expect(genes[i]).to be <= 1

        changed_x = true if genes[i] != genes_x[i]
        changed_y = true if genes[i] != genes_y[i]
      end

      expect(changed_x).to be(true)
      expect(changed_y).to be(true)
    end
  end

  describe 'mutate' do
    it 'random' do
      genes = 100.times.map{ rand(-1.0..1.0) }

      new_genes = Ai::Neat.mutate(:random, genes, 0.5)

      changed = false

      (0..(genes.count - 1)).each do |i|
        expect(genes[i]).to be >= -1
        expect(genes[i]).to be <= 1

        changed = true if genes[i] != new_genes[i]
      end

      expect(changed).to be(true)
    end

    it 'edit' do
      genes = 100.times.map{ rand(-1.0..1.0) }

      new_genes = Ai::Neat.mutate(:edit, genes, 0.5)

      changed = false

      (0..(genes.count - 1)).each do |i|
        expect(genes[i]).to be >= -1
        expect(genes[i]).to be <= 1

        changed = true if genes[i] != new_genes[i]
      end

      expect(changed).to be(true)
    end
  end
end