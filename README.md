# F1Tenth @ UCI Workshop Repo

# Introduction

We add [DreamerV3](https://arxiv.org/abs/2301.04104) implementations of World Models into the F1tenth ROS2 environment in PyTorch. Referenced [dreamerv3-torch](https://github.com/NM512/dreamerv3-torch), [racing_dreamer](https://github.com/CPS-TUWien/racing_dreamer), and [dreamerv3](https://github.com/danijar/dreamerv3) repositories for the base code.

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please reference: [CONTRIBUTING.md](CONTRIBUTING.md) for detailed contribution guide.

## Testing

- [x] implemented f1tenth gym for testing environment.
- [ ] waiting on issac sim implementation for better simulation environment.

## Checklist:

- [x] basic structure implementation(bare bones)
- [ ] LiDAR feed for encoded latent state
- [ ] World Model implementation
- [ ] Actor-Critic implementation
- [ ] Scaling reward system
