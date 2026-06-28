import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import lightning as pl
import h5py

class CustomDataset(Dataset):
    def __init__(self, h5_file, group_size):
        """
        Initializes the dataset by setting the HDF5 file path and group size.
        
        Args:
            h5_file (str): Path to the HDF5 file.
            group_size (int): Size of each style for content data.(For example, group_size: 7000 means 7000 content images are used for training from one style folder.") 
        """
        self.h5_file = h5_file
        self.group_size = group_size
        self.h5f = None  
        
        # Open the file temporarily to get the total number of rows and image featuresS
        with h5py.File(self.h5_file, 'r') as h5f:
            self.total_rows = h5f['images'].shape[0]
            self.data = np.array(h5f['images'], dtype=np.float32)

        
        self.num_groups = self.total_rows // self.group_size
    def __len__(self):
        return self.total_rows


    def __getitem__(self, idx):
        # Check if idx is a tuple of indices (for triplet sampling)
        if isinstance(idx, tuple):
            # Extract embeddings for each index in the triplet
            selected_data = np.array([self.data[i] for i in idx])
            embeddings = torch.tensor(selected_data, dtype=torch.float32)
            indices = torch.tensor(idx, dtype=torch.long)
            return embeddings, indices
        else:
            # Standard single index behavior
            embedding = torch.tensor(self.data[idx], dtype=torch.float32)
            index = torch.tensor(idx, dtype=torch.long)
            return embedding, index


class FixTripletSampler(torch.utils.data.Sampler):
    def __init__(self, num_groups, group_size, num_samples,seed=None, current_rank=0, world_size=1, style_idx=-1, content_idx=-1):
        self.num_groups = num_groups
        self.group_size = group_size
        self.num_samples = num_samples
        self.current_rank = current_rank
        self.world_size = world_size
        self.style_idx = style_idx
        self.content_idx = content_idx
        self.generator = torch.Generator()
        if seed is not None:
            self.generator.manual_seed(seed)        
        self.sample_indices = self._generate_samples()

    def _generate_samples(self):

        group_idx = torch.randint(0, self.num_groups, (self.num_samples,), generator=self.generator)
        if self.style_idx != -1:
            idx = self.style_idx #8 #27 #11 #9  #8
            group_idx = torch.full((self.num_samples,), idx, dtype=torch.long)
        elif self.num_samples == self.num_groups:
            group_idx = torch.randperm(self.num_groups)
        local_indices = torch.randint(0, self.group_size, (self.num_samples, 2), generator=self.generator)

        random_offsets = torch.randint(1, self.num_groups, (self.num_samples,), generator=self.generator)
        other_group_idx = (group_idx + random_offsets) % self.num_groups

        embed1_idx = group_idx * self.group_size + local_indices[:, 0]
        if self.content_idx != -1:
            local_indices[:, 1] = self.content_idx
        elif self.num_samples == 100:
            local_indices[:, 1] = torch.randperm(self.group_size)[:100]
        embed2_idx = group_idx * self.group_size + local_indices[:, 1]
        embed3_idx = other_group_idx * self.group_size + local_indices[:, 1]

        indices = list(zip(embed1_idx.tolist(), embed2_idx.tolist(), embed3_idx.tolist()))
        return indices


    def __iter__(self):
        return iter(self.sample_indices)
    def __len__(self):
        return self.num_samples
        
class TripletSampler(torch.utils.data.Sampler):
    def __init__(self, num_groups, group_size, num_samples,seed=None, current_rank=0, world_size=1):
        self.num_groups = num_groups
        self.group_size = group_size
        self.num_samples = num_samples
        self.seed = seed 
        self.current_rank = current_rank
        self.world_size = world_size
        self.generator = torch.Generator()  
        self.generator.manual_seed(self.seed)     
        print(f"DynamicTripletSampler Initialized - Num Groups: {self.num_groups}, Group Size: {self.group_size}, Num Samples: {self.num_samples}")
        
    def __iter__(self):
        # Generate random group -> style_1
        group_idx = torch.randint(0, self.num_groups, (self.num_samples,), generator=self.generator)
        # local_indices -> content
        local_indices = torch.randint(0, self.group_size, (self.num_samples, 2), generator=self.generator)
        # style_2
        random_offsets = torch.randint(1, self.num_groups, (self.num_samples,), generator=self.generator)
        other_group_idx = (group_idx + random_offsets) % self.num_groups

        embed1_idx = group_idx * self.group_size + local_indices[:, 0]
        embed2_idx = group_idx * self.group_size + local_indices[:, 1]
        embed3_idx = other_group_idx * self.group_size + local_indices[:, 1]

        indices = list(zip(embed1_idx.tolist(), embed2_idx.tolist(), embed3_idx.tolist()))

        return iter(indices) 

    def __len__(self):
        return self.num_samples

    
def triplet_collate_fn(batch, group_size=None, include_labels=False):
    embeddings, indices = zip(*batch)
    embeddings = torch.stack(embeddings, dim=0)  # Shape: (batch_size, 3, embedding_dim)
    indices = torch.stack(indices, dim=0)        # Shape: (batch_size, 3)  # Collect indices

    if include_labels:
        if group_size is None:
            raise ValueError("group_size is required when include_labels=True")
        style_labels = indices // group_size
        content_labels = indices % group_size
        return embeddings, indices, style_labels.long(), content_labels.long()

    return embeddings, indices
  

class CustomDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_npy, val_npy,num_workers= 0, val_num_workers= 0, train_group_size=700, val_group_size=300,batch_size=4,val_batch_size=4,batches_per_epoch=20000, val_n_samples=256, seed=42, style_idx=-1, content_idx=-1, include_labels=False
    ):
        super().__init__()
        self.train_npy = train_npy
        self.val_npy = val_npy
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.train_group_size = train_group_size
        self.val_group_size = val_group_size
        self.num_workers = num_workers
        self.val_num_workers = val_num_workers
        self.batches_per_epoch = batches_per_epoch
        self.val_n_samples = val_n_samples
        self.seed = seed
        self.style_idx = style_idx
        self.content_idx = content_idx
        self.include_labels = include_labels

    def setup(self, stage=None):
        self.train_dataset = CustomDataset(self.train_npy, self.train_group_size)
        self.val_dataset = CustomDataset(self.val_npy, self.val_group_size)
        
        if self.trainer is None:
            self.train_seed = self.seed + 0
            self.val_seed = self.seed + 10000
        else:
            self.train_seed = self.seed + self.trainer.global_rank
            self.val_seed = self.seed + self.trainer.global_rank + 10000
        self.world_size = self.trainer.world_size

        self.train_sampler = TripletSampler(
            num_groups=self.train_dataset.num_groups,
            group_size=self.train_group_size,
            num_samples=self.batches_per_epoch*self.batch_size*self.world_size,
            seed = self.train_seed,
            current_rank=self.trainer.global_rank,
            world_size=self.world_size
        )
        self.val_sampler = FixTripletSampler(
            num_groups=self.val_dataset.num_groups,
            group_size=self.val_group_size,
            num_samples=self.val_n_samples,
            seed = self.val_seed,
            current_rank=self.trainer.global_rank,
            world_size=self.world_size
        )
        

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler= self.train_sampler,
            num_workers=self.num_workers,
            collate_fn=lambda batch: triplet_collate_fn(
                batch, group_size=self.train_group_size, include_labels=self.include_labels
            )
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            sampler=self.val_sampler,
            num_workers=self.val_num_workers,
            collate_fn=lambda batch: triplet_collate_fn(
                batch, group_size=self.val_group_size, include_labels=self.include_labels
            )
        )
    def test_dataloader(self):
        self.val_dataset = CustomDataset(self.val_npy, self.val_group_size)
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            sampler=FixTripletSampler(
            num_groups=self.val_dataset.num_groups,
            group_size=self.val_group_size,
            num_samples=self.val_n_samples,
            seed = self.seed,
            style_idx=self.style_idx,
            content_idx=self.content_idx
        ),
            num_workers=self.val_num_workers,
            collate_fn=lambda batch: triplet_collate_fn(
                batch, group_size=self.val_group_size, include_labels=self.include_labels
            )
        )
