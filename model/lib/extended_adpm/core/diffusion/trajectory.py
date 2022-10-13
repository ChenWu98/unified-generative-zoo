import math
import numpy as np


def _choice_steps_linear(N, sample_steps):
    assert sample_steps > 1
    frac_stride = (N - 1) / (sample_steps - 1)
    cur_idx = 1.0
    steps = []
    for _ in range(sample_steps):
        steps.append(round(cur_idx))
        cur_idx += frac_stride
    return steps


def _choice_steps_linear_ddim(N, sample_steps):
    skip = N // sample_steps
    seq = list(range(1, N + 1, skip))
    return seq


def _choice_steps_quad_ddim(N, sample_steps):
    seq = np.linspace(0, np.sqrt(N * 0.8), sample_steps) ** 2
    seq = [int(s) + 1 for s in list(seq)]
    return seq


def _round_and_remove_dup(seq):
    seq = [round(item) for item in seq]
    val_old = -float('inf')
    for idx, val in enumerate(seq):
        if val <= val_old:
            seq[idx] = int(val_old) + 1
        val_old = seq[idx]
    return seq


def _choice_steps_rfn(N, sample_steps, rfn):  # rfn: reverse of a function fn, s.t., fn(0)=1 and fn(1)=0
    assert sample_steps > 1
    ys = [k / (sample_steps - 1) for k in range(sample_steps)][::-1]
    xs = [rfn(y) for y in ys]
    assert xs[0] == 0 and xs[-1] == 1
    steps = [(N - 1) * x + 1 for x in xs]
    assert steps[0] == 1 and steps[-1] == N
    steps = _round_and_remove_dup(steps)
    assert steps[0] == 1 and steps[-1] == N
    assert all(steps[i] < steps[i+1] for i in range(len(steps) - 1))
    return steps


def _split(ms_eps, N, K):
    idx_g1 = N + 1
    for n in range(1, N + 1):  # Theoretically, ms_eps <= 1. Remove points of poor estimation
        if ms_eps[n] > 1:
            idx_g1 = n
            break
    num_bad = 2 * (N - idx_g1 + 1)
    bad_ratio = num_bad / N

    N1 = N - num_bad
    K1 = math.ceil((1. - 0.8 * bad_ratio) * K)
    K2 = K - K1
    if K1 > N1:
        K1 = N1
        K2 = K - K1
    if K2 > num_bad:
        K2 = num_bad
        K1 = K - K2
    if num_bad > 0 and K2 == 0:
        K2 = 1
        K1 = K - K2
    assert num_bad <= N
    assert K1 <= N1 and K2 <= N - N1
    return K1, N1, K2, num_bad


def _ms_score(ms_eps, betas):
    alphas = 1. - betas
    cum_alphas = alphas.cumprod()
    cum_betas = 1. - cum_alphas
    ms_score = np.zeros_like(ms_eps)
    ms_score[1:] = ms_eps[1:] / cum_betas[1:]
    return ms_score


def _solve_fn_dp(fn, N, K):  # F[st, ed] with 1 <= st < ed <= N, other elements is inf
    if N == K:
        return list(range(1, N + 1))

    F = fn[: N + 1, : N + 1]

    C = np.full((K + 1, N + 1), float('inf'))  # C[k, n] with 2 <= k <= K, k <= n <= N
    D = np.full((K + 1, N + 1), -1)  # D[k, n] with 2 <= k <= K, k <= n <= N

    C[2, 2: N] = F[1, 2: N]
    D[2, 2: N] = 1

    for k in range(3, K + 1):
        # {C[k-1, s] + F[s, r]}_{0 <= s, r <= N} = {C[k-1, s] + F[s, r]}_{k-1 <= s < r <= N}
        tmp = C[k - 1, :].reshape(N + 1, 1) + F
        C[k, k: N + 1] = np.min(tmp, axis=0)[k: N + 1]
        D[k, k: N + 1] = np.argmin(tmp, axis=0)[k: N + 1]

    res = [N]
    n, k = N, K
    while k > 2:
        n = D[k, n]
        res.append(n)
        k -= 1
    res.append(1)
    return res[::-1]


def _solve_fn_dp_general(fn, a, b, K):  # F[st, ed] with a <= st < ed <= b, other elements is inf
    N = b - a + 1
    F = np.full((N + 1, N + 1), float('inf'))  # F[st, ed] with 1 <= st < ed <= N
    F[1: N + 1, 1: N + 1] = fn[a: b + 1, a: b + 1]
    res = _solve_fn_dp(F, N, K)
    return [idx + a - 1 for idx in res]


def _get_fn_m(ms_score, alphas, N):
    F = np.full((N + 1, N + 1), float('inf'))  # F[st, ed] with 1 <= st < ed <= N
    for s in range(1, N + 1):
        skip_alphas = alphas[s + 1: N + 1].cumprod()
        skip_betas = 1. - skip_alphas
        before_log = 1. - skip_betas * ms_score[s + 1: N + 1]
        F[s, s + 1: N + 1] = np.log(before_log)
    return F


def _dp_seg(ms_eps, betas, N, K):
    K1, N1, K2, num_bad = _split(ms_eps, N, K)

    alphas = 1. - betas
    ms_score = _ms_score(ms_eps, betas)
    F = _get_fn_m(ms_score, alphas, N1)

    steps1 = _solve_fn_dp(F, N1, K1)
    if K2 > 0:
        frac = (N - N1) / K2
        steps2 = [round(N - frac * k) for k in range(K2)][::-1]
        assert steps1[-1] < steps2[0]
        assert len(steps1) + len(steps2) == K
        assert steps1[0] == 1 and steps1[-1] == N1
        assert steps2[-1] == N
    else:
        steps2 = []
    steps = steps1 + steps2
    assert steps[0] == 1 and steps[-1] == N
    assert all(steps[i] < steps[i + 1] for i in range(len(steps) - 1))
    return steps


################################################################################
# for dp_vb
################################################################################
def make_inf(F):  # make F[s, t] = inf for s >= t
    return np.triu(F, 1) + np.tril(np.full(F.shape, float('inf')))


cached_F, cached_N, cached_D = None, None, None


def vectorized_dp(F, N):  # F[s, t] with 0 <= s < t <= N
    global cached_F, cached_N, cached_D

    if cached_F is not None and cached_N is not None and (F == cached_F).all() and N == cached_N:
        return cached_D

    F = make_inf(F[: N + 1, : N + 1])

    C = np.full((N + 1, N + 1), float('inf'))
    D = np.full((N + 1, N + 1), -1)

    C[0, 0] = 0
    for k in range(1, N + 1):
        bpds = C[k - 1, :].reshape(N + 1, 1) + F
        C[k] = np.min(bpds, axis=0)
        D[k] = np.argmin(bpds, axis=0)

    cached_F, cached_N, cached_D = F, N, D
    return D


def fetch_path(D, N, K):  # find a path of length K (K+1 nodes)
    optpath = []
    t = N
    for k in reversed(range(K + 1)):
        optpath.append(t)
        t = D[k, t]
    return optpath[::-1]


def _choice_steps(N, sample_steps, typ, ms_eps=None, nll_terms=None, betas=None):
    if typ == 'linear':
        steps = _choice_steps_linear(N, sample_steps)
    elif typ.startswith('power'):
        power = int(typ.split('power')[1])
        steps = _choice_steps_rfn(N, sample_steps, rfn=lambda y: 1 - y ** (1. / power))
    elif typ == 'linear_ddim':
        steps = _choice_steps_linear_ddim(N, sample_steps)
    elif typ == 'quad_ddim':
        steps = _choice_steps_quad_ddim(N, sample_steps)
    elif typ == 'dp_seg':
        steps = _dp_seg(ms_eps, betas, N, sample_steps)
    elif typ == 'dp_vb':
        F = nll_terms['F']  # should be train nll terms
        assert len(F) - 1 == N
        D = vectorized_dp(F, N)
        path = fetch_path(D, N, sample_steps)
        assert path[0] == 0
        steps = path[1:]
    else:
        raise NotImplementedError

    assert len(steps) == sample_steps
    if typ != 'dp_vb':
        assert steps[0] == 1
    if typ not in ["linear_ddim", "quad_ddim"]:
        assert steps[-1] == N

    return steps
