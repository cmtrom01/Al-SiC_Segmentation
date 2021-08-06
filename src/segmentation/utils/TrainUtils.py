import numpy as np
import torch

class TrainUtils:

	def __init__(self):
		self.smooth = 0.6

	def dice_no_threshold(self, outputs: torch.Tensor, targets: torch.Tensor, eps: float = 1e-7, threshold: float = 0.3, activation: str = "Sigmoid"):

		#activation_fn = get_activation_fn(activation)
		#outputs = activation_fn(outputs)
		outputs = torch.sigmoid(outputs)

		if threshold is not None:
			outputs = (outputs > threshold).float()

		intersection = torch.sum(targets * outputs)
		union = torch.sum(targets) + torch.sum(outputs)
		dice = 2 * intersection / (union + eps)

		return dice
