from datasets import load_dataset
from torch.utils.data import DataLoader

class HuggingFaceInstructions:
    
    def __init__(self,):
        self.dataset = load_dataset("HuggingFaceH4/no_robots")
        self.data_loader = DataLoader(
            self.dataset, batch_size=4, shuffle=True
        )