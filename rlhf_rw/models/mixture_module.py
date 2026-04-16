import numpy as np
import torch
import torch.nn as nn

try:
    from sklearn.mixture import GaussianMixture
    _SKLEARN_OK = True
except ImportError:
    _SKLEARN_OK = False


def _descriptor_size(K, F, cov_type):
    pi_dim = K
    mu_dim = K * F
    if cov_type == "full":
        cov_dim = K * (F * (F + 1) // 2)
    elif cov_type == "diag":
        cov_dim = K * F
    elif cov_type == "spherical":
        cov_dim = K
    elif cov_type == "tied":
        cov_dim = F * (F + 1) // 2
    else:
        raise ValueError(f"unknown cov_type: {cov_type}")
    return pi_dim + mu_dim + cov_dim


def _flatten_gmm(gmm, K, F, cov_type):
    weights = gmm.weights_
    means = gmm.means_
    covs = gmm.covariances_

    order = np.argsort(-weights)
    weights = weights[order]
    means = means[order]

    pi_part = weights.astype(np.float32)
    mu_part = means.flatten().astype(np.float32)

    if cov_type == "full":
        covs = covs[order]
        iu = np.triu_indices(F)
        cov_part = np.stack([c[iu] for c in covs]).flatten().astype(np.float32)
    elif cov_type == "diag":
        covs = covs[order]
        cov_part = covs.flatten().astype(np.float32)
    elif cov_type == "spherical":
        covs = covs[order]
        cov_part = covs.flatten().astype(np.float32)
    elif cov_type == "tied":
        iu = np.triu_indices(F)
        cov_part = covs[iu].astype(np.float32)
    else:
        raise ValueError(cov_type)

    return np.concatenate([pi_part, mu_part, cov_part])


def _zero_descriptor(K, F, cov_type):
    return np.zeros(_descriptor_size(K, F, cov_type), dtype=np.float32)


def _fit_one(features, K, F, cov_type, reg_covar=1e-4, seed=0):
    if not _SKLEARN_OK:
        return _zero_descriptor(K, F, cov_type)
    if features.size == 0:
        return _zero_descriptor(K, F, cov_type)

    norms = np.linalg.norm(features, axis=1)
    features = features[norms > 1e-6]

    n = len(features)
    if n < max(K * 2, 4):
        return _zero_descriptor(K, F, cov_type)

    effective_K = K
    if cov_type == "full":
        params_per_comp = F + F * (F + 1) // 2
        max_K = max(1, n // (params_per_comp + 1))
        effective_K = min(K, max_K)

    try:
        gmm = GaussianMixture(
            n_components=effective_K,
            covariance_type=cov_type,
            random_state=seed,
            reg_covar=reg_covar,
            max_iter=100,
            n_init=1,
            init_params="k-means++",
        )
        gmm.fit(features)
    except Exception:
        return _zero_descriptor(K, F, cov_type)

    if effective_K < K:
        full_weights = np.zeros(K, dtype=np.float32)
        full_means = np.zeros((K, F), dtype=np.float32)
        full_weights[:effective_K] = gmm.weights_
        full_means[:effective_K] = gmm.means_

        if cov_type == "full":
            full_covs = np.tile(np.eye(F)[None] * 1e-2, (K, 1, 1)).astype(np.float32)
            full_covs[:effective_K] = gmm.covariances_
        elif cov_type == "diag":
            full_covs = np.ones((K, F), dtype=np.float32) * 1e-2
            full_covs[:effective_K] = gmm.covariances_
        elif cov_type == "spherical":
            full_covs = np.ones(K, dtype=np.float32) * 1e-2
            full_covs[:effective_K] = gmm.covariances_
        else:
            full_covs = gmm.covariances_

        class _Stub:
            pass
        s = _Stub()
        s.weights_ = full_weights
        s.means_ = full_means
        s.covariances_ = full_covs
        return _flatten_gmm(s, K, F, cov_type)

    return _flatten_gmm(gmm, K, F, cov_type)


class MixtureTokenModule(nn.Module):
    """Per-response GMM summary token.

    Forward:
        fixations:  (B, S, F)  ET feature tensor
        attn_mask:  (B, S)     1=real token, 0=padding (optional)

    Returns:
        token: (B, 1, hidden_size)
        mask:  (B, 1)
    """

    def __init__(
        self,
        num_features=5,
        K=3,
        hidden_size=4096,
        cov_type="diag",
        proj_hidden=128,
        dropout=0.1,
        log_transform=True,
    ):
        super().__init__()
        self.F = num_features
        self.K = K
        self.cov_type = cov_type
        self.log_transform = log_transform
        self.desc_size = _descriptor_size(K, num_features, cov_type)

        self.projector = nn.Sequential(
            nn.Linear(self.desc_size + 1, proj_hidden),
            nn.LayerNorm(proj_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(proj_hidden, hidden_size),
            nn.LayerNorm(hidden_size),
        )

    def _compute_descriptors(self, fixations, attn_mask=None):
        B = fixations.shape[0]
        S = fixations.shape[1]
        descriptors = np.zeros((B, self.desc_size + 1), dtype=np.float32)
        fix_np = fixations.detach().float().cpu().numpy()
        if attn_mask is not None:
            mask_np = attn_mask.detach().cpu().numpy().astype(bool)
            if mask_np.ndim == 1:
                mask_np = mask_np[None, :]
        else:
            mask_np = np.ones(fix_np.shape[:2], dtype=bool)

        for b in range(B):
            row_fix = fix_np[b]
            row_mask = mask_np[b] if b < mask_np.shape[0] else np.ones(S, dtype=bool)
            # Length mismatch safety: crop both to the common min length.
            # Upstream pad/remap can leave fixations and attention mask at
            # different sequence lengths; rather than crash, just take the
            # overlapping prefix.
            n = min(row_fix.shape[0], row_mask.shape[0])
            feats = row_fix[:n][row_mask[:n]]
            if self.log_transform:
                feats = np.log1p(np.clip(feats, 0, None))
            try:
                desc = _fit_one(feats, self.K, self.F, self.cov_type)
                valid = 1.0 if np.any(np.abs(desc) > 1e-8) else 0.0
                descriptors[b, :-1] = desc
                descriptors[b, -1] = valid
            except Exception:
                descriptors[b, -1] = 0.0

        return torch.from_numpy(descriptors).to(fixations.device)

    def forward(self, fixations, attn_mask=None):
        desc = self._compute_descriptors(fixations, attn_mask)
        desc = desc.to(next(self.projector.parameters()).dtype)
        token_emb = self.projector(desc).unsqueeze(1)
        token_mask = torch.ones(
            token_emb.shape[0], 1, dtype=torch.long, device=token_emb.device
        )
        return token_emb, token_mask
