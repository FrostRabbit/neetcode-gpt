import torch
import torch.nn as nn
from typing import List, Dict


class Solution:

    def compute_activation_stats(self, model: nn.Module, x: torch.Tensor) -> List[Dict[str, float]]:
        # Forward pass through model layer by layer
        # After each nn.Linear, record: mean, std, dead_fraction
        # Run with torch.no_grad(). Round to 4 decimals.
        result = []
        with torch.no_grad():
            for module in model.children():
                x = module(x)
                if isinstance(module, nn.Linear):
                    m = round(x.mean().item(), 4)
                    std = round(x.std().item(),4)
                    d = (x<=0).all(dim=0)
                    if x.size(1) > 0:
                        dead_frac = round(d[d==True].nelement()/x.size(1),4)
                    else: dead_frac = 0.0
                    result.append(
                        {
                            'mean': m,
                            'std': std,
                            'dead_fraction': dead_frac
                        }
                    )
        return result

    def compute_gradient_stats(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> List[Dict[str, float]]:
        # Forward + backward pass with nn.MSELoss
        # For each nn.Linear layer's weight gradient, record: mean, std, norm
        # Call model.zero_grad() first. Round to 4 decimals.
        model.zero_grad()
        pred = model(x)
        loss_func = nn.MSELoss()
        loss = loss_func(pred, y)
        loss.backward()
        result = []
        for module in model.children():
            if isinstance(module, nn.Linear):
                grad = module.weight.grad
                m = round(grad.mean().item(),4)
                std = round(grad.std().item(),4)
                norm = round(torch.norm(grad).item(), 4)
                result.append(
                    {
                        'mean': m,
                        'std': std,
                        'norm': norm
                    }
                )
        return result

    def diagnose(self, activation_stats: List[Dict[str, float]], gradient_stats: List[Dict[str, float]]) -> str:
        # Classify network health based on the stats
        # Return: 'dead_neurons', 'exploding_gradients', 'vanishing_gradients', or 'healthy'
        # Check in priority order (see problem description for thresholds)
        for i in activation_stats:
            if i['dead_fraction'] > 0.5: return 'dead_neurons'
        for i in gradient_stats:  
            if i['norm'] > 1000.0: return 'exploding_gradients'
        
        if gradient_stats[-1]['norm'] < 1e-5: return 'vanishing_gradients'
        for i in activation_stats:    
            if i['std'] < 0.1: return 'vanishing_gradients'
            if i['std'] > 10.0: return 'exploding_gradients'
        return 'healthy'
