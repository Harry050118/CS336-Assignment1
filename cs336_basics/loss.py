import torch

def cross_entropy(logits: torch.Tensor, labels: torch.Tensor):
    '''
    loss = -mean(log(softmax(logits)))
         = -mean((logits - log(sum(exp(logits)))))

    args:
        logits:  (batch_size, vocab_size) unnormalized logits
        targets: (batch_size,) integer class indices
    returns:
        scalar: average cross entropy loss over all examples
    '''

    # Subtract the largest element for numerical stability.
    logits = logits - torch.max(logits, dim=1, keepdim=True).values

    # (batch_size, vocab_size) -> (batch_size, vocab_size) 
    log_probs = logits - torch.log(torch.sum(torch.exp(logits), dim=1, keepdim=True))

    labels = labels.unsqueeze(1)                       # (batch_size, ) -> (batch_size, 1)
    loss = log_probs.gather(1, labels)                 # 每一行取 labels 指定的位置
    loss = loss.squeeze(1)                             # (batch_size, 1) -> (batch_size, )             
    loss = -loss.mean()
    return loss