import torch

def Stealthy_Shield_Defense(
    f: torch.Tensor,  # [batch_size, class_num]
    Dʼ: torch.Tensor,  # [class_num, class_num]
    T=0.1,
    ε=0.5,
) -> torch.Tensor:  # [batch_size, class_num]
    P = (f / T).softmax(dim=1)
    y = torch.multinomial(P, 1).view(-1)
    q = Dʼ[y]
    H = -(f * f.log()).sum(dim=1)
    return GPU_based_water_filling(f, q, ε * H)

def GPU_based_water_filling(
    f: torch.Tensor,  # [batch_size, class_num]
    q: torch.Tensor,  # [batch_size, class_num]
    ε: torch.Tensor,  # [batch_size]
) -> torch.Tensor:  # [batch_size, class_num]
    m = (f - q).norm(dim=1, p=1) <= ε

    W, index = (f / q).sort()
    fʼ = f.gather(dim=1, index=index)
    qʼ = q.gather(dim=1, index=index)
    row = torch.arange(len(W))

    F = fʼ.cumsum(dim=1)
    Q = qʼ.cumsum(dim=1)
    M = W[:, 1:] * Q[:, :-1] - F[:, :-1] <= ε.view(-1, 1) / 2
    j = M.int().argmin(dim=1)
    w_A = (F[row, j] + ε / 2) / Q[row, j]

    F = fʼ.flip(-1).cumsum(dim=1).flip(-1)
    Q = qʼ.flip(-1).cumsum(dim=1).flip(-1)
    M = W[:, :-1] * Q[:, 1:] - F[:, 1:] >= -ε.view(-1, 1) / 2
    M = torch.cat([M, M.new_ones(len(M), 1)], dim=1)
    j = M.int().argmax(dim=1)
    w_B = (F[row, j] - ε / 2) / Q[row, j]

    m, w_A, w_B = m.view(-1, 1), w_A.view(-1, 1), w_B.view(-1, 1)
    return q.where(m, f.clip(w_A * q, w_B * q))
