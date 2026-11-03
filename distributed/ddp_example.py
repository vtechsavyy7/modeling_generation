import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

# 1. Simple Dummy Dataset
class ToyDataset(Dataset):
    def __init__(self, size=256):
        self.data = torch.randn(size, 10)
        self.labels = torch.randn(size, 1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# 2. Main Training Function
def main():
    # Initialize the default process group using environment variables set by torchrun
    # "nccl" is the recommended backend for NVIDIA GPUs
    dist.init_process_group(backend="gloo")
    
    # Retrieve the local rank assigned to this specific process/GPU
    # NOTE: For now, setting all devices to cuda:0 since this machine has only 1 GPU
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:0")
    torch.cuda.set_device(device)
    

    # Initialize Dataset and DistributedSampler
    # The sampler ensures each GPU sees a non-overlapping portion of the data
    dataset = ToyDataset()
    sampler = DistributedSampler(dataset, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=16, sampler=sampler)

    # Instantiate model and move it to the specific GPU
    model = nn.Linear(10, 1).to(device)
    
    # Wrap the model with DDP
    # Because multiple processes are sharing one GPU, you cannot bind 
    # the process exclusively to a device ID. DDP will use the current device context.
    ddp_model = DDP(model)

    criterion = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    # Simple training loop (1 epoch)
    # Important: Set the epoch on the sampler for proper shuffling across nodes
    sampler.set_epoch(0)
    
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = ddp_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # Only print or save checkpoints on the master process (rank 0) to avoid duplicates
    if dist.get_rank() == 0:
        print("Training finished safely on rank 0!")

    # Clean up the process group
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
