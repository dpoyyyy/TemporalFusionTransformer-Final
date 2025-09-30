# Temporal Fusion Transformer (TFT) in PyTorch

This repository contains a clean, architecture-only PyTorch implementation of the **Temporal Fusion Transformer (TFT)**, a powerful and interpretable model for multi-horizon time series forecasting. The implementation is designed to be modular and easy to integrate into larger projects.

This code is based on the original paper: [Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting](https://arxiv.org/abs/1912.09363) by Lim, Arık, Loeff, & Pfister.

## About This Implementation

Please note that this `main.py` file contains **only the model's architecture**. The runnable examples and testing scripts have been intentionally excluded to provide a clean, import-ready module. A detailed usage example is provided below in this README.

## Key Features

  - **Modular Design:** The architecture is broken down into its core components (`GRN`, `VariableSelectionNetwork`, `InterpretableMultiHeadAttention`, etc.), making it easy to understand and extend.
  - **Faithful to the Paper:** Implements key mechanisms described in the TFT paper, including Gated Residual Networks, Variable Selection, and interpretable multi-head attention.
  - **Minimal Dependencies:** Requires only PyTorch, making it lightweight and easy to set up.

## Architecture Overview

The model is constructed from several building blocks, each implemented as a `torch.nn.Module`:

  - **`GatedResidualNetwork (GRN)`:** The core building block used throughout the model to apply non-linear transformations with residual connections.
  - **`VariableSelectionNetwork (VSN)`:** Learns to select the most relevant input variables at each time step, enhancing interpretability.
  - **`StaticCovariateEncoder`:** Encodes static (time-invariant) metadata into context vectors.
  - **`Seq2SeqLSTM`:** A sequence-to-sequence layer for processing local temporal patterns in past and future data.
  - **`InterpretableMultiHeadAttention`:** A modified multi-head attention mechanism that provides interpretable attention weights across the time series.
  - **`TemporalFusionTransformer`:** The final integrated model that combines all the components above to produce forecasts.

## Contributing

Contributions are welcome\! If you find a bug or have a suggestion for improvement, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
