"""Differentiable PyTorch wrapper around a Qibo state-vector QCNN."""

import math

import numpy as np
import torch
from torch import nn
from qibo.models import Circuit

if __package__:
    from .data_encoding import add_angle_encoding
    from .pooling import add_pooling_layer
    from .qcnn_circuit import add_conv_layer
else:
    from data_encoding import add_angle_encoding
    from pooling import add_pooling_layer
    from qcnn_circuit import add_conv_layer


def _expectation_z(state: np.ndarray, n_qubits: int, qubit: int) -> float:
    probabilities = np.abs(np.asarray(state).reshape(-1)) ** 2
    basis_states = np.arange(probabilities.size)
    bit_position = n_qubits - 1 - qubit
    signs = 1.0 - 2.0 * ((basis_states >> bit_position) & 1)
    return float(np.dot(probabilities, signs).real)


def _simulate_qcnn(
    features: np.ndarray,
    weights: np.ndarray,
    n_qubits: int,
    n_outputs: int,
    shifted_feature: tuple[int, float] | None = None,
    shifted_weight: tuple[int, int, int, float] | None = None,
) -> np.ndarray:
    circuit = Circuit(n_qubits)
    add_angle_encoding(circuit, features, shifted_feature)
    active_qubits = list(range(n_qubits))

    for layer_index, layer_weights in enumerate(weights):
        layer_shift = None
        if shifted_weight is not None and shifted_weight[0] == layer_index:
            _, pair_index, weight_index, shift = shifted_weight
            layer_shift = (pair_index, weight_index, shift)
        add_conv_layer(circuit, active_qubits, layer_weights, layer_shift)

        if len(active_qubits) > n_outputs:
            keep = active_qubits[::2]
            discard = active_qubits[1::2]
            add_pooling_layer(circuit, keep, discard)
            active_qubits = keep

    state = circuit.execute().state()
    return np.asarray(
        [_expectation_z(state, n_qubits, qubit) for qubit in active_qubits],
        dtype=float,
    )


class _ParameterShiftQCNN(torch.autograd.Function):
    @staticmethod
    def forward(ctx, features, weights, n_qubits, n_outputs):
        ctx.n_qubits = n_qubits
        ctx.n_outputs = n_outputs
        ctx.save_for_backward(features, weights)

        feature_values = features.detach().numpy()
        weight_values = weights.detach().numpy()
        output = np.stack(
            [
                _simulate_qcnn(
                    sample,
                    weight_values,
                    n_qubits,
                    n_outputs,
                )
                for sample in feature_values
            ]
        )
        return torch.as_tensor(output, dtype=features.dtype, device=features.device)

    @staticmethod
    def backward(ctx, grad_output):
        features, weights = ctx.saved_tensors
        feature_values = features.detach().numpy()
        weight_values = weights.detach().numpy()
        output_gradient = grad_output.detach().numpy()
        feature_gradient = np.zeros_like(feature_values, dtype=float)
        weight_gradient = np.zeros_like(weight_values, dtype=float)
        shift = np.pi / 2.0

        for batch_index, sample in enumerate(feature_values):
            upstream_gradient = output_gradient[batch_index]
            if ctx.needs_input_grad[0]:
                for feature_index in range(ctx.n_qubits):
                    positive = _simulate_qcnn(
                        sample,
                        weight_values,
                        ctx.n_qubits,
                        ctx.n_outputs,
                        shifted_feature=(feature_index, shift),
                    )
                    negative = _simulate_qcnn(
                        sample,
                        weight_values,
                        ctx.n_qubits,
                        ctx.n_outputs,
                        shifted_feature=(feature_index, -shift),
                    )
                    derivative = 0.5 * (positive - negative) * np.pi
                    feature_gradient[batch_index, feature_index] = np.dot(
                        upstream_gradient, derivative
                    )

            if ctx.needs_input_grad[1]:
                active_count = ctx.n_qubits
                for layer_index in range(weight_values.shape[0]):
                    for pair_index in range(active_count - 1):
                        for weight_index in range(weight_values.shape[1]):
                            positive = _simulate_qcnn(
                                sample,
                                weight_values,
                                ctx.n_qubits,
                                ctx.n_outputs,
                                shifted_weight=(
                                    layer_index,
                                    pair_index,
                                    weight_index,
                                    shift,
                                ),
                            )
                            negative = _simulate_qcnn(
                                sample,
                                weight_values,
                                ctx.n_qubits,
                                ctx.n_outputs,
                                shifted_weight=(
                                    layer_index,
                                    pair_index,
                                    weight_index,
                                    -shift,
                                ),
                            )
                            derivative = 0.5 * (positive - negative)
                            weight_gradient[layer_index, weight_index] += np.dot(
                                upstream_gradient, derivative
                            )
                    active_count //= 2

        return (
            torch.as_tensor(feature_gradient, dtype=features.dtype)
            if ctx.needs_input_grad[0]
            else None,
            torch.as_tensor(weight_gradient, dtype=weights.dtype)
            if ctx.needs_input_grad[1]
            else None,
            None,
            None,
        )


class QCNNModel(nn.Module):
    """A binary/multi-output QCNN with parameter-shift differentiation.

    Inputs must have shape ``(batch, n_qubits)`` and be normalized to [0, 1].
    The model returns Z expectation values in [-1, 1] for the surviving qubits.
    Qibo's state-vector simulator runs on CPU; gradients are computed exactly
    with the parameter-shift rule, including shared convolution-gate instances.
    """

    def __init__(self, n_qubits: int, n_outputs: int = 1):
        super().__init__()
        if not isinstance(n_qubits, int) or n_qubits < 2:
            raise ValueError("n_qubits must be an integer of at least 2.")
        if n_qubits & (n_qubits - 1):
            raise ValueError("n_qubits must be a power of two for this pooling layout.")
        if not isinstance(n_outputs, int) or n_outputs < 1:
            raise ValueError("n_outputs must be a positive integer.")
        if n_outputs >= n_qubits or n_outputs & (n_outputs - 1):
            raise ValueError("n_outputs must be a power of two smaller than n_qubits.")

        layer_ratio = n_qubits // n_outputs
        if n_qubits % n_outputs or layer_ratio & (layer_ratio - 1):
            raise ValueError("n_qubits / n_outputs must be a power of two.")

        self.n_qubits = n_qubits
        self.n_outputs = n_outputs
        n_layers = int(math.log2(layer_ratio))
        self.weights = nn.Parameter(torch.empty(n_layers, 4))
        nn.init.uniform_(self.weights, -0.25, 0.25)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if not isinstance(features, torch.Tensor):
            raise TypeError("features must be a PyTorch tensor.")
        if features.device.type != "cpu":
            raise ValueError("the Qibo state-vector backend currently supports CPU tensors only.")
        if not features.is_floating_point():
            raise TypeError("features must use a floating-point dtype.")
        if features.ndim != 2 or features.shape[1] != self.n_qubits:
            raise ValueError(
                f"features must have shape (batch, {self.n_qubits})."
            )
        if features.shape[0] == 0:
            raise ValueError("features must contain at least one sample.")
        if not torch.isfinite(features).all():
            raise ValueError("features must contain only finite values.")
        if torch.any((features < 0.0) | (features > 1.0)):
            raise ValueError("features must be normalized to the [0, 1] interval.")
        return _ParameterShiftQCNN.apply(
            features.contiguous(),
            self.weights,
            self.n_qubits,
            self.n_outputs,
        )


if __name__ == "__main__":
    sample_batch = torch.tensor([[0.1, 0.5, 0.9, 0.2]], dtype=torch.float32)
    model = QCNNModel(n_qubits=4)
    prediction = model(sample_batch)
    loss = prediction.square().mean()
    loss.backward()
    print("Z expectations:", prediction.detach())
    print("Weight gradients:", model.weights.grad)
