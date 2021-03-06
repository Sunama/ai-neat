# Ai::Neat

This gem is base on [NEAT-JS](https://github.com/ExtensionShoe/NEAT-JS). That have the main purpose to convert the code from javascript to ruby.

## Installation

Add this line to your application's Gemfile:

```ruby
gem 'ai-neat'
```

And then execute:

    $ bundle install

Or install it yourself as:

    $ gem install ai-neat

## Usage

1. Get started with config

```
config = {
  models: [
    { node_count: 5, node_type: :input },
    { node_count: 3, node_type: :output, activationfunc: :softmax }
  ],
  mutation_rate: 0.1,
  crossover_method: :random,
  mutation_method: :random,
  population_size: population_size
}
```

`models` parameter are array of layers of NEAT at least their are needed 2 layers are `node_type` = `:input` and `:output` but it's possible to add any `:middle` layer between of them.

```
  models: [
    { node_count: 5, node_type: :input },
    { node_count: 5, node_type: :inpu, activationfunc: :softmax },
    { node_count: 3, node_type: :output, activationfunc: :softmax }
  ]
```

And `activationfunc` are needed if that layer is not output layer. Current supported `activationfunc` are `:relu`, `:tanh`, `:sigmoid`, `:leaky_relu`, `:softmax`.

Supported `crossover_method` are `:random` and `:slice`.

Supported `mutation_method` are `:random` and `:edit`.

2. Initiate `Neat` Class from previous config

```
neat = Ai::Neat::Neat.new(config)
```

3. Set generation loop

4. Set inputs array equal to number of `node_type` = `:input` to each Creature `[i]`

```
neat.set_inputs(inputs, i)
```

If your simmulation have various creature life time like a gameover you can skip to set inputs that creature.

5. When set inputs compleated

```
neat.feed_forward
```

6. Get decisions for each creature in array

```
decisions = neat.decisions
```

returned `decisions` value are array of integer that started from `0` to `node_count - 1` of `:output` layer.

7. Finish generation

```
neat.do_gen
```

And you can get best creature result by

```
neat.best_creature
```

This function will return index of best creature in lastest generation.

## Example

```
population_size = 100

config = {
models: [
    { node_count: 5, node_type: :input },
    { node_count: 3, node_type: :output, activationfunc: :softmax }
],
mutation_rate: 0.1,
crossover_method: :random,
mutation_method: :random,
population_size: population_size
}

neat = Ai::Neat::Neat.new(config)

scores = population_size.times.map { 0 }
first_score = 0

# Generation
100.times.each do |gen|
scores = population_size.times.map { 0 }

# play
50.times.each do
    inputs = 5.times.map { rand(-1.0..1.0) }

    (0..(scores.count - 1)).each do |i|
        neat.set_inputs(inputs, i)
    end

    neat.feed_forward
    decisions = neat.decisions

    (0..(scores.count - 1)).each do |i|
        if inputs.last > 0
            case decisions[i]
            when 0
            scores[i] += 1
            when 1
            scores[i] -= 1
            end
        else
            case decisions[i]
            when 0
            scores[i] -= 1
            when 1
            scores[i] += 1
            end
        end

        scores[i] = 0 if scores[i] < 0
    end
end

(0..(scores.count - 1)).each do |i|
    neat.set_fitness(scores[i], i)
end

first_score = scores[neat.best_creature] if gen == 0

neat.do_gen
end

exported = neat.export

neat2 = Ai::Neat::Neat.new(config)
neat2.import(exported)
```

## Development

After checking out the repo, run `bin/setup` to install dependencies. Then, run `rake spec` to run the tests. You can also run `bin/console` for an interactive prompt that will allow you to experiment.

To install this gem onto your local machine, run `bundle exec rake install`. To release a new version, update the version number in `version.rb`, and then run `bundle exec rake release`, which will create a git tag for the version, push git commits and the created tag, and push the `.gem` file to [rubygems.org](https://rubygems.org).

## Contributing

Bug reports and pull requests are welcome on GitHub at https://github.com/Sunama/ai-neat. This project is intended to be a safe, welcoming space for collaboration, and contributors are expected to adhere to the [code of conduct](https://github.com/Sunama/ai-neat/blob/master/CODE_OF_CONDUCT.md).

## License

The gem is available as open source under the terms of the [MIT License](https://opensource.org/licenses/MIT).

## Code of Conduct

Everyone interacting in the Ai::Neat project's codebases, issue trackers, chat rooms and mailing lists is expected to follow the [code of conduct](https://github.com/Sunama/ai-neat/blob/master/CODE_OF_CONDUCT.md).
