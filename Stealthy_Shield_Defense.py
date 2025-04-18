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
    f_old: torch.Tensor,  # [batch_size, class_num]
    q_old: torch.Tensor,  # [batch_size, class_num]
    ε: torch.Tensor,  # [batch_size]
) -> torch.Tensor:  # [batch_size, class_num]
    m = (f_old - q_old).norm(dim=1, p=1) <= ε

    W, index = (f_old / q_old).sort()
    f = f_old.gather(dim=1, index=index)
    q = q_old.gather(dim=1, index=index)
    row = torch.arange(len(W))

    F = f.cumsum(dim=1)
    Q = q.cumsum(dim=1)
    M = W[:, 1:] * Q[:, :-1] - F[:, :-1] <= ε.view(-1, 1) / 2
    j = M.int().argmin(dim=1)
    w_A = (F[row, j] + ε / 2) / Q[row, j]

    F = f.flip(-1).cumsum(dim=1).flip(-1)
    Q = q.flip(-1).cumsum(dim=1).flip(-1)
    M = W[:, :-1] * Q[:, 1:] - F[:, 1:] >= -ε.view(-1, 1) / 2
    M = torch.cat([M, M.new_ones(len(M), 1)], dim=1)
    j = M.int().argmax(dim=1)
    w_B = (F[row, j] - ε / 2) / Q[row, j]

    m, w_A, w_B = m.view(-1, 1), w_A.view(-1, 1), w_B.view(-1, 1)
    return q_old.where(m, f_old.clip(w_A * q_old, w_B * q_old))
