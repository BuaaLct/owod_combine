
import torch

alpha =1

def get_cluster_prob(embeddings,centers):
    norm_squared = torch.sum((embeddings.unsqueeze(1) - centers) ** 2, 2)
    numerator = 1.0 / (1.0 + (norm_squared / alpha))
    power = float(alpha + 1) / 2
    numerator = numerator ** power
    return numerator / torch.sum(numerator, dim=1, keepdim=True)
