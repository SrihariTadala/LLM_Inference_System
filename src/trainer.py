from typing import Any, Dict
from os import path

import torch
from torch.nn import functional as F
from torch.nn import utils
from torch.utils.data import DataLoader


class Trainer:
	def __init__(self, model, tokenizer, optimizer, train_dataloader: DataLoader, **kwargs: Dict[str, Any]):
		self.model = model
		self.tokenizer = tokenizer
		self.optimizer = optimizer
		self.train_dataloader = train_dataloader
		self.model_save_path = path.join(kwargs['model_save_dir'], kwargs['ckpt'])
		self.kwargs = kwargs

	def train(self):
		epoch, min_loss = self.kwargs['epoch'], float("inf")
		for e in range(1, epoch + 1):
			train_loss = self._train_step()
			print(f"Train Loss after {e} epoch: {train_loss}")
			if min_loss > train_loss:
				min_loss = train_loss
				self.model.save_ckpt(e, self.model_save_path, self.optimizer)

	def train_ddp(self):
		device = self.kwargs['device']
		epoch, min_loss = self.kwargs['epoch'], float("inf")
		for e in range(1, epoch + 1):
			self.train_dataloader.sampler.set_epoch(e)
			train_loss = self._train_step()
			if device == 0:
				print(f"Train Loss after {e} epoch: {train_loss}")
				if min_loss > train_loss:
					min_loss = train_loss
					self.model.module.save_ckpt(e, self.model_save_path, self.optimizer)

	def _train_step(self) -> float:
		device = self.kwargs.get('device')
		device_type = self.kwargs.get('device_type')
		use_mix_precision = self.kwargs.get('use_mix_precision', True)
		gradient_clip = self.kwargs.get('gradient_clip', 1.0)
		running_loss = 0.0
		for input_token_ids, target_token_ids in self.train_dataloader:
			input_token_ids, target_token_ids = input_token_ids.to(device), target_token_ids.to(device)
			# Clearing gradients from last run.
			self.optimizer.zero_grad()
			with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=use_mix_precision):
				predictions = self.model(input_token_ids)
				loss = F.cross_entropy(
					predictions.view(-1, predictions.shape[-1]), target_token_ids.view(-1), 
					ignore_index=self.tokenizer.pad_token_id
				)
			# Calculte loss backwards and update model weights.
			loss.backward()
			# Clipping Gradients.
			utils.clip_grad_norm_(self.model.parameters(), max_norm=gradient_clip)
			self.optimizer.step()
			running_loss += loss.item()

		return running_loss / len(self.train_dataloader)
