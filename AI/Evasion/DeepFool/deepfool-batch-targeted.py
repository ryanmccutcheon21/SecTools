import os
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
    load_model,
)

# Normalize inputs to match training preprocessing
MNIST_MEAN, MNIST_STD = 0.1307, 0.3081

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