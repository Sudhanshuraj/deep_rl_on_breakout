Usage
--------

If you're working on OpenAI's [Breakout-v4](https://gym.openai.com/envs/Breakout-v4/) environment:
 * To train: `python baby-a3c.py --env Breakout-v4`
 * To test: `python baby-a3c.py --env Breakout-v4 --test True`
 * To render: `python baby-a3c.py --env Breakout-v4 --render True`


Architecture
--------

```python
self.linear1 = nn.Linear(input_shape, output_shape1)
self.linear2 = nn.Linear(output_shape1, output_shape2)
self.gru = nn.GRUCell(output_shape2, memsize)
self.critic_linear, self.actor_linear = nn.Linear(memsize, 1), nn.Linear(memsize, num_actions)
```
