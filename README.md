
### Toolpath Design for Additive Manufacturing using Reinforcement Learning

Using Muzero algorithm to design toolpaths for additive manufacturing process given any section geometry. Currently, we use a dense reward structure where a positive reward is assigned to correct material deposition, a negative reward is assigned to wrong material deposition, and a small negative reward is assigned to other motions at each time step. Significant parts of this code are adopted based on [muzero-general](https://github.com/werner-duvaud/muzero-general).

## To Dos
- [x] Accelerate the code
- [x] Dense reward structure test
- [ ] Sparse reward structure test
- [ ] Document hyper-parameter analysis
- [ ] Pretraining networks

## Getting started
### Installation

```
git clone https://github.com/mojtabamozaffar/toolpath-design-rl
cd toolpath-design-rl

pip install -r requirements.txt
```

### Usage
```
python main.py
```

## License

This project is released under the [MIT License](https://github.com/mojtabamozaffar/toolpath-design-rl/blob/master/LICENSE).
