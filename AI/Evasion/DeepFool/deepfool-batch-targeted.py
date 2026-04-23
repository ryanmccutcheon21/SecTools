# Check for virtual environments and set up pip install
import os
import subprocess
import sys
import argparse, json, time
from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import io, base64, requests, random
from PIL import Image

from htb_ai_library import (
    set_reproducibility,
    MNISTClassifierWithDropout,
    get_mnist_loaders,
    train_model,
    evaluate_accuracy,
    save_model,
    load_model,
    analyze_model_confidence,
    HTB_GREEN, NODE_BLACK, HACKER_GREY, WHITE,
    AZURE, NUGGET_YELLOW, MALWARE_RED, VIVID_PURPLE, AQUAMARINE
)

# Normalize inputs to match training preprocessing
MNIST_MEAN, MNIST_STD = 0.1307, 0.3081

base64_images = [
"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"
]

class SimpleClassifier(nn.Module):     
	"""CNN matching the server-side architecture with log-softmax outputs."""     
	
	def __init__(self) -> None:        
		super().__init__()        
		self.conv1 = nn.Conv2d(1, 32, 3, 1)        
		self.conv2 = nn.Conv2d(32, 64, 3, 1)        
		self.dropout1 = nn.Dropout(0.25)        
		self.dropout2 = nn.Dropout(0.5)        
		self.fc1 = nn.Linear(9216, 128)        
		self.fc2 = nn.Linear(128, 10)     
		
	def forward(self, x: torch.Tensor) -> torch.Tensor:        
		x = self.conv1(x)        
		x = torch.relu(x)        
		x = self.conv2(x)        
		x = torch.relu(x)        
		x = torch.max_pool2d(x, 2)        
		x = self.dropout1(x)        
		x = torch.flatten(x, 1)        
		x = self.fc1(x)        
		x = torch.relu(x)        
		x = self.dropout2(x)        
		x = self.fc2(x)        
		return torch.log_softmax(x, dim=1)

def mnist_normalize(x01: torch.Tensor) -> torch.Tensor:     
	"""Normalize a [0,1] tensor to MNIST stats for the classifier."""    
	return (x01 - MNIST_MEAN) / MNIST_STD

def png_from_x01(x01: np.ndarray) -> str:     
	"""Encode a `[0,1]` grayscale image `(28,28)` to base64 PNG string."""    
	x255 = np.clip((x01 * 255.0).round(), 0, 255).astype(np.uint8)    
	img = Image.fromarray(x255, mode="L")    
	buf = io.BytesIO()    
	img.save(buf, format="PNG", optimize=True)    
	return base64.b64encode(buf.getvalue()).decode("ascii") 
	
def png_from_x_anysize(x01: np.ndarray, size: tuple[int, int]) -> str:     
	"""Encode a `[0,1]` grayscale array to base64 PNG of a specific size.     
	Parameters    ----------    
	x01 : np.ndarray        
		Input 2D array in `[0,1]`.    
	size : (int, int)        
		Target `(width, height)` for the PNG.    
	"""    
	x255 = np.clip((x01 * 255.0).round(), 0, 255).astype(np.uint8)    
	img = Image.fromarray(x255, mode="L").resize(size, resample=Image.NEAREST)    
	buf = io.BytesIO()    
	img.save(buf, format="PNG", optimize=True)    
	return base64.b64encode(buf.getvalue()).decode("ascii") 
	
def x01_from_b64_png(b64: str) -> np.ndarray:     
	"""Decode base64 PNG to `[0,1]` numpy array of shape `(28,28)`."""    
	raw = base64.b64decode(b64)    
	img = Image.open(io.BytesIO(raw)).convert("L")    
	if img.size != (28, 28):        
		raise ValueError("Expected 28x28 PNG")    
	x = np.asarray(img, dtype=np.float32) / 255.0    
	return np.clip(x, 0.0, 1.0)

@dataclass 
class Challenge:     
	l2_threshold: float    
	target: int    
	label: int    
	sample_index: int    
	x01: np.ndarray  # (1,1,28,28)
 
#def fetch_challenge(host: str, retries: int = 30, delay: float = 1.0) -> Challenge:     
#	"""Fetch challenge with simple retry/backoff to tolerate startup races."""    
#    for b64_img in base64_images: 
#	    x2d = x01_from_b64_png(b64_img)  # (28,28)    
#	    x4d = x2d[None, None, ...]    
#	return Challenge(        
#	    l2_threshold=float(payload["l2_threshold"]),        
#	    target=int(payload["target"]),        
#	    label=int(payload["label"]),        
#	    sample_index=int(payload["sample_index"]),        
#	    x01=x4d.astype(np.float32),    
#	   )
 
def load_model(weights_path: str) -> SimpleClassifier:     
	model = SimpleClassifier()    
	state = torch.load(weights_path, map_location=torch.device("cpu"))    
	model.load_state_dict(state)    
	model.eval()    
	return model


def deepfool_targeted(     
	model: nn.Module,    
	x01: np.ndarray,    
	target: int,    
	overshoot: float = 0.08,    
	max_iter: int = 100, 
) -> np.ndarray:     
	"""Compute a targeted DeepFool adversarial example in `[0,1]` pixel space.     
	
	The update follows the linearized boundary between the current predicted    class and the fixed `target` class, stepping by the minimal L2 amount    required to cross that boundary, with a small overshoot. Iterates are    clamped to `[0,1]` to preserve a valid image domain.     
	
	Parameters    ----------    
	model : nn.Module        
		Classifier in eval mode.    
	x01 : np.ndarray        
		Baseline `[0,1]` image with shape `(1,1,28,28)`.    
	target : int        
		Desired target class in `[0, 9]`.    
	overshoot : float, optional        
		Multiplicative margin used on the accumulated perturbation to remain        across the decision boundary after PNG quantization, by default 0.08.    
	max_iter : int, optional        
		Maximum number of iterations, by default 100.     
		
	Returns    -------    
	np.ndarray        
		Adversarial image in `[0,1]` with shape `(1,1,28,28)`.    
	"""    
	x01_t = torch.from_numpy(x01).float()    
	r_tot = torch.zeros_like(x01_t)    
	with torch.enable_grad():        
		for _ in range(max_iter):            
			x = (                
				torch.clamp(x01_t + (1 + overshoot) * r_tot, 0.0, 1.0)                
				.detach()                
				.requires_grad_(True)            
			)            
			logits = model(mnist_normalize(x))            
			pred = int(torch.argmax(logits, dim=1).item())            
			if pred == target:                
				break            
			# Gradients for current prediction and target            
			model.zero_grad(set_to_none=True)            
			logits[0, pred].backward(retain_graph=True)            
			grad_pred = x.grad.detach().clone()            
			x.grad.zero_()            
			model.zero_grad(set_to_none=True)            
			logits[0, target].backward(retain_graph=True)            
			grad_t = x.grad.detach().clone()            
			x.grad.zero_()            
			w = grad_t - grad_pred            
			g = (logits[0, target] - logits[0, pred]).detach()            
			denom = torch.norm(w.flatten()) + 1e-12            
			p = torch.abs(g) / denom            
			r_i = (p + 1e-4) * w / (torch.norm(w.flatten()) + 1e-12)            
			r_tot = r_tot + r_i    
	x_adv = torch.clamp(x01_t + (1 + overshoot) * r_tot, 0.0, 1.0)    
	return x_adv.detach().cpu().numpy()

def main() -> None:     
	parser = argparse.ArgumentParser()    
	parser.add_argument(        
		"--host", default="http://127.0.0.1:8000", help="Server base URL"    
	)    
	parser.add_argument(        
		"--weights", default="solver/deepfool_weights.pth", help="Path to model weights (downloads from /weights if missing)"    
	)    
	args = parser.parse_args()     
	
	chall = fetch_challenge(args.host)    
	if not os.path.exists(args.weights):        
		os.makedirs(os.path.dirname(args.weights), exist_ok=True)        
		wb = requests.get(f"{args.host}/weights", timeout=15).content        
		with open(args.weights, "wb") as f:            
			f.write(wb)    
	model = load_model(args.weights)     
	
	# Local sanity prediction on clean image    
	x = torch.from_numpy(chall.x01)    
	clean_pred = int(torch.argmax(model(mnist_normalize(x)), dim=1).item())    
	if clean_pred != chall.label:        
		print(            
			f"Warning: local clean prediction {clean_pred} != label {chall.label}; proceeding regardless"        
		)     
		
	# Run targeted DeepFool with a robust overshoot schedule against PNG quantization    
	overshoots = [0.08, 0.10, 0.12, 0.15, 0.25]    
	x_adv = None    
	adv_pred = None    
	l2 = None    
	for ov in overshoots:        
		cand = deepfool_targeted(            
			model, chall.x01, target=chall.target, overshoot=ov, max_iter=100        
		)        
		cand_pred = int(            
			torch.argmax(model(mnist_normalize(torch.from_numpy(cand))), dim=1).item()        
		)        
		cand_l2 = float(np.linalg.norm(cand - chall.x01))        
		if cand_pred == chall.target and cand_l2 <= chall.l2_threshold:            
			x_adv, adv_pred, l2 = cand, cand_pred, cand_l2            
			break    
	if x_adv is None:        
		# Take the closest candidate and proceed; server will validate        
		cand = deepfool_targeted(            
			model, chall.x01, target=chall.target, overshoot=overshoots[0], max_iter=100        
		)        
		x_adv = cand        
		adv_pred = int(            
			torch.argmax(model(mnist_normalize(torch.from_numpy(cand))), dim=1).item()        
		)        
		l2 = float(np.linalg.norm(cand - chall.x01))    
	print(        
		json.dumps(            
			{                
				"l2": l2,                
				"pred": adv_pred,                
				"clean_pred": clean_pred,                
				"threshold": chall.l2_threshold,                
				"target": chall.target,            
			},            
			indent=2,        
		)    
	)     
	
	# Submit    
	b64 = png_from_x01(x_adv[0, 0])    
	r = requests.post(f"{args.host}/submit", json={"image_b64": b64}, timeout=15)    
	try:        
		r.raise_for_status()    
	except Exception:        
		print("Server response:", r.text)        
		raise    
	print("Flag:", r.json().get("flag"))     
	
	# Robustness tests (negative cases)    
	print("\nRobustness checks (expected rejections):")     
	
	def attempt(desc: str, payload: dict | None, path: str = "/submit") -> dict:        
		try:            
			if payload is None:                
				resp = requests.post(f"{args.host}{path}", json={}, timeout=10)            
			else:                
				resp = requests.post(f"{args.host}{path}", json=payload, timeout=10)            
			status = resp.status_code            
			detail = None            
			try:                
				detail = resp.json()            
			except Exception:                
				detail = {"text": resp.text}            
			return {"case": desc, "status": status, "detail": detail}        
		except Exception as e:  # noqa: BLE001            
			return {"case": desc, "status": None, "error": str(e)}     
			
	base2d = chall.x01[0, 0]    
	cases = []     
	
	# 1) Clean image (should fail: not the target class)    
	cases.append(        
		attempt(            
			"clean-image",            
			{"image_b64": png_from_x01(base2d)},        
		)    
	)     
	
	# 2) Too large L2 (zeros image), likely misclassified but exceeds threshold    
	zeros2d = np.zeros_like(base2d, dtype=np.float32)    
	cases.append(        
		attempt(            
			"l2-too-large",            
			{"image_b64": png_from_x01(zeros2d)},        
		)    
	)     
	
	# 3) Wrong image size (32x32)    
	cases.append(        
		attempt(            
			"wrong-size-32x32",            
			{"image_b64": png_from_x_anysize(base2d, (32, 32))},        
		)    
	)     
	
	# 4) Invalid base64/PNG    
	cases.append(        
		attempt(            
			"invalid-base64",            
			{"image_b64": "not_base64_png_data"},        
		)    
	)     
	
	# 5) Missing field (schema error)    
	cases.append(attempt("missing-field", None))     
	
	# 6) Predict endpoint with wrong size    
	cases.append(        
		attempt(            
			"predict-wrong-size",            
			{"image_b64": png_from_x_anysize(base2d, (30, 30))},            path="/predict",        
		)    
	)     
	
	# 7) Adversarial but wrong target (should fail targeted check while within L2)    
	def craft_wrong_target() -> dict | None:        
		# Try all non-target classes and several overshoots; pick first that meets L2 and predicts that class        
		classes = [c for c in range(10) if c != chall.target]        
		for c in classes:            
			for ov in [0.08, 0.10, 0.12, 0.15, 0.25]:                
				cand = deepfool_targeted(                    
					model, chall.x01, target=c, overshoot=ov, max_iter=100                
				)                
				# Quantization round-trip to match server path                
				b64 = png_from_x01(cand[0, 0])                
				x2d = x01_from_b64_png(b64)                
				cand_q = x2d[None, None, ...].astype(np.float32)                
				pred = int(                    
					torch.argmax(                        model(mnist_normalize(torch.from_numpy(cand_q))), dim=1                    ).item()                
				)                
				l2_q = float(np.linalg.norm(cand_q - chall.x01))                
				if pred == c and l2_q <= chall.l2_threshold:                    
					return {                        
						"image_b64": b64,                        
						"meta": {"class": c, "l2": l2_q, "overshoot": ov},                    
					}        
		return None     
	wrong = craft_wrong_target()    
	if wrong is not None:        
		cases.append(            
			attempt("adversarial-wrong-target", {"image_b64": wrong["image_b64"]})        
		)    
	else:        
		cases.append(            
			{                
				"case": "adversarial-wrong-target",                
				"status": None,                
				"detail": {"note": "could not construct within L2"},            
			}        
		)     
		
	print(json.dumps({"negative_cases": cases}, indent=2)) 
	
if __name__ == "__main__":     
	main()