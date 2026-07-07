# Architecture and figure artifacts

Generated artifacts:

## Matplotlib numeric figures
- `fig_125m_best_val_loss_by_seed.pdf/png`
- `fig_125m_muon_delta_by_seed.pdf/png`
- `fig_125m_validation_curves_by_seed.pdf/png`
- `fig_125m_quality_speed_tradeoff.pdf/png`
- `fig_scaling_summary_57m_to_125m.pdf/png`
- `fig_minigpt_125m_3d_stack.pdf/png`

## TikZ / PGFPlots
- `tikz/fig_minigpt_architecture_tikz.tex/pdf`
- `pgfplots/fig_125m_best_val_loss_pgfplots.tex/pdf`

## PlotNeuralNet
- `plotneuralnet/fig_plotneuralnet_minigpt_125m.tex/pdf`
- vendor source: `vendor/PlotNeuralNet/`

## SVG
- `svg/fig_minigpt_architecture.svg`

## Netron
- `netron/minigpt_125m_schematic.onnx`
- Open with: `python -m netron figures/netron/minigpt_125m_schematic.onnx`

## TensorBoard
- `tensorboard/minigpt_125m_graph/`
- Open with: `tensorboard --logdir figures/tensorboard`

## VisualKeras
- `visualkeras/fig_visualkeras_minigpt_125m.png`
- This is a conceptual Keras-backed view of the PyTorch architecture, not the benchmark training model.
