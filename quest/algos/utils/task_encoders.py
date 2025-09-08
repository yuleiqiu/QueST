"""
This file defines a PyTorch module for encoding task IDs into dense vector representations.
Note: this file is not used in current experiments.
"""
import torch.nn as nn

class TaskEmbeddingEncoder(nn.Module):
    """
    A module to encode a task ID into a dense vector representation (embedding).
    This is useful in multi-task learning scenarios where the model needs to be
    conditioned on the specific task it is performing.
    """
    def __init__(self, n_tasks, embed_dim):
        """
        Initializes the TaskEmbeddingEncoder.

        Args:
            n_tasks (int): The total number of distinct tasks.
            embed_dim (int): The dimensionality of the embedding vector for each task.
        """
        super().__init__()
        # Create an embedding layer that acts as a lookup table.
        # It stores `n_tasks` embedding vectors, each of size `embed_dim`.
        self.task_encodings = nn.Embedding(n_tasks, embed_dim)

    def forward(self, task_id):
        """
        Performs the forward pass of the encoder.

        Args:
            task_id (torch.Tensor): A tensor containing the ID(s) of the task(s).

        Returns:
            torch.Tensor: The embedding vector(s) corresponding to the input task ID(s).
        """
        # Look up the embedding vector for the given task_id.
        return self.task_encodings(task_id)
