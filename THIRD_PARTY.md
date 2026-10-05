# References and attribution

The primary sources below were inspected on 2026-10-01 before implementing
the JEPA components. This implementation uses the task's scaled dimensions
and cart-pole adaptations, and does not claim to reproduce the published model.

- **LeWorldModel**, Lucas Maes, Quentin Le Lidec, Damien Scieur, Yann LeCun,
  Randall Balestriero: [paper v1](https://arxiv.org/html/2603.19312v1),
  especially Section 3.1 and Appendices A and D.
- **Official LeWorldModel implementation**, commit
  `8edfeb336732b5f3ce7b8b210d0ba370a09e2cac`:
  [module.py](https://github.com/lucas-maes/le-wm/blob/8edfeb336732b5f3ce7b8b210d0ba370a09e2cac/module.py)
  and [train.py](https://github.com/lucas-maes/le-wm/blob/8edfeb336732b5f3ce7b8b210d0ba370a09e2cac/train.py).
  `losses.py` adapts the small SIGReg implementation; `models.py` adapts the
  token-wise AdaLN-zero pattern. Copyright (c) 2026 Lucas Maes, MIT License;
  the complete notice is retained in [licenses/LeWorldModel-MIT.txt](licenses/LeWorldModel-MIT.txt).
  Changes include batch-first trajectory layout, a separate reproducible
  projection generator, float32 enforcement, native PyTorch attention, explicit
  three-token windows, and task-specific architecture and conditioning.
  The exact reference quadrature uses 17 knots on [0,3], a Gaussian integration
  window, doubled trapezoid weights, and a multiplier equal to the number of
  independent trajectories. These implementation conventions take precedence
  over the paper appendix's illustrative integration interval.
- **LeJEPA minimal implementation**, Randall Balestriero and Yann LeCun:
  [MINIMAL.md](https://github.com/galilai-group/lejepa/blob/c293d291ca87cd4fddee9d3fffe4e914c7272052/MINIMAL.md),
  commit `c293d291ca87cd4fddee9d3fffe4e914c7272052`, inspected for the
  characteristic-function and symmetric quadrature conventions. Its repository
  is CC BY-NC 4.0; no code is copied from it. The reused code comes from the
  MIT-licensed LeWorldModel source above.
- The encoder imports torchvision's unpretrained ResNet-18 implementation;
  torchvision is a dependency rather than vendored source.
