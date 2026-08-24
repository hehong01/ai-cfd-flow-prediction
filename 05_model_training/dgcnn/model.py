"""
DGCNN regression model for CFD surface-field prediction.

Input per point:
    [x, y, z, velocity]

Output per point:
    [HTC, wall_shear, pressure]

Expected input:
    (B, N, 4)

Expected output:
    (B, N, 3)

Main structure:
    EdgeConv 1 : 4   -> 64
    EdgeConv 2 : 64  -> 64
    EdgeConv 3 : 64  -> 128

    Local feature concatenation:
        64 + 64 + 128 = 256

    Global feature:
        max pooling over all N points
        256 -> 256

    Local + global:
        256 + 256 = 512

    Regression head:
        512 -> 256 -> 128 -> 3

Dynamic graph:
    k-NN is recomputed before every EdgeConv using the current
    point features.

For N=7000, exact k-NN is calculated in query chunks to avoid
constructing the full NxN distance matrix at once.
"""

from __future__ import annotations

import torch
import torch.nn as nn


# =====================================================================
# Exact chunked k-NN
# =====================================================================

@torch.no_grad()
def exact_knn_chunked(
    features: torch.Tensor,
    k: int = 20,
    chunk_size: int = 1024,
) -> torch.Tensor:
    """
    Find exact k-nearest neighbors for every point.

    Parameters
    ----------
    features:
        Shape:
            (B, N, F)

    k:
        Number of nearest neighbors.

    chunk_size:
        Number of query points processed at once.

    Returns
    -------
    neighbor_indices:
        Shape:
            (B, N, k)

    Notes
    -----
    Neighbor selection is discrete and does not require gradients.

    k-NN indices are calculated using detached features.

    The indices are then used to gather from the original feature
    tensor inside EdgeConv, so gradients still flow through the
    neural-network operations.
    """

    if features.ndim != 3:
        raise ValueError(
            f"Expected features shape (B, N, F), "
            f"found {tuple(features.shape)}."
        )

    batch_size, num_points, feature_dim = (
        features.shape
    )

    if feature_dim <= 0:
        raise ValueError(
            "Feature dimension must be positive."
        )

    if k <= 0:
        raise ValueError(
            "k must be positive."
        )

    if k >= num_points:
        raise ValueError(
            f"k must be smaller than number of points. "
            f"k={k}, N={num_points}"
        )

    if chunk_size <= 0:
        raise ValueError(
            "chunk_size must be positive."
        )

    # -------------------------------------------------------------
    # k-NN graph construction does not require gradient.
    # Use float32 for distance calculation.
    # -------------------------------------------------------------

    graph_features = (
        features
        .detach()
        .float()
    )

    reference = graph_features

    # (B, F, N)
    reference_t = reference.transpose(
        1,
        2,
    )

    # ||x_j||^2
    #
    # (B, 1, N)
    reference_squared = (
        reference
        .pow(2)
        .sum(
            dim=-1,
        )
        .unsqueeze(1)
    )

    neighbor_chunks = []

    # =============================================================
    # Query points are processed in chunks.
    # =============================================================

    for start in range(
        0,
        num_points,
        chunk_size,
    ):

        end = min(
            start + chunk_size,
            num_points,
        )

        # (B, Q, F)
        query = reference[
            :,
            start:end,
            :,
        ]

        # ||x_i||^2
        #
        # (B, Q, 1)
        query_squared = (
            query
            .pow(2)
            .sum(
                dim=-1,
                keepdim=True,
            )
        )

        # ---------------------------------------------------------
        # Squared Euclidean distance:
        #
        # ||a-b||^2
        # =
        # ||a||^2 + ||b||^2 - 2a·b
        #
        # Shape:
        #     (B, Q, N)
        # ---------------------------------------------------------

        distances = (
            query_squared
            + reference_squared
            - 2.0
            * torch.bmm(
                query,
                reference_t,
            )
        )

        # Numerical round-off can produce tiny negative values.
        distances.clamp_min_(
            0.0
        )

        # ---------------------------------------------------------
        # Exclude each query point itself.
        # ---------------------------------------------------------

        query_count = (
            end - start
        )

        row_indices = torch.arange(
            query_count,
            device=features.device,
        )

        point_indices = torch.arange(
            start,
            end,
            device=features.device,
        )

        distances[
            :,
            row_indices,
            point_indices,
        ] = float("inf")

        # ---------------------------------------------------------
        # Exact k nearest neighbors.
        # ---------------------------------------------------------

        indices = torch.topk(
            distances,
            k=k,
            dim=-1,
            largest=False,
            sorted=False,
        ).indices

        neighbor_chunks.append(
            indices
        )

    neighbor_indices = torch.cat(
        neighbor_chunks,
        dim=1,
    )

    expected_shape = (
        batch_size,
        num_points,
        k,
    )

    if neighbor_indices.shape != expected_shape:
        raise RuntimeError(
            "Unexpected k-NN output shape: "
            f"{tuple(neighbor_indices.shape)}"
        )

    return neighbor_indices


# =====================================================================
# Edge feature construction
# =====================================================================

def build_edge_features(
    features: torch.Tensor,
    neighbor_indices: torch.Tensor,
) -> torch.Tensor:
    """
    Construct DGCNN edge features:

        [f_i, f_j - f_i]

    Parameters
    ----------
    features:
        Shape:
            (B, N, F)

    neighbor_indices:
        Shape:
            (B, N, k)

    Returns
    -------
    edge_features:
        Shape:
            (B, N, k, 2F)
    """

    if features.ndim != 3:
        raise ValueError(
            f"Expected features shape (B, N, F), "
            f"found {tuple(features.shape)}."
        )

    if neighbor_indices.ndim != 3:
        raise ValueError(
            f"Expected neighbor_indices shape (B, N, k), "
            f"found {tuple(neighbor_indices.shape)}."
        )

    batch_size, num_points, feature_dim = (
        features.shape
    )

    index_batch_size, index_num_points, k = (
        neighbor_indices.shape
    )

    if (
        index_batch_size != batch_size
        or index_num_points != num_points
    ):
        raise ValueError(
            "Feature and neighbor-index shapes "
            "are incompatible."
        )

    # -------------------------------------------------------------
    # Convert per-batch point indices into indices for a flattened
    # (B*N, F) representation.
    # -------------------------------------------------------------

    batch_offsets = (
        torch.arange(
            batch_size,
            device=features.device,
        )
        .view(
            batch_size,
            1,
            1,
        )
        * num_points
    )

    flat_indices = (
        neighbor_indices
        + batch_offsets
    )

    flat_features = features.reshape(
        batch_size * num_points,
        feature_dim,
    )

    # -------------------------------------------------------------
    # Neighbor features:
    #
    # (B, N, k, F)
    # -------------------------------------------------------------

    neighbor_features = flat_features[
        flat_indices.reshape(-1)
    ].reshape(
        batch_size,
        num_points,
        k,
        feature_dim,
    )

    # -------------------------------------------------------------
    # Center features:
    #
    # (B, N, k, F)
    # -------------------------------------------------------------

    center_features = (
        features
        .unsqueeze(2)
        .expand(
            -1,
            -1,
            k,
            -1,
        )
    )

    # -------------------------------------------------------------
    # Relative feature:
    #
    # f_j - f_i
    # -------------------------------------------------------------

    relative_features = (
        neighbor_features
        - center_features
    )

    # -------------------------------------------------------------
    # Edge feature:
    #
    # [f_i, f_j - f_i]
    #
    # (B, N, k, 2F)
    # -------------------------------------------------------------

    edge_features = torch.cat(
        (
            center_features,
            relative_features,
        ),
        dim=-1,
    )

    return edge_features


# =====================================================================
# EdgeConv
# =====================================================================

class EdgeConv(nn.Module):
    """
    One Dynamic EdgeConv block.

    Workflow:

        point features
            ↓
        exact k-NN
            ↓
        [f_i, f_j - f_i]
            ↓
        shared Edge MLP
            ↓
        one feature vector per edge
            ↓
        max over k neighbors
            ↓
        new point feature

    The Edge MLP is deliberately kept simple:

        Linear(2 * input_dim, output_dim)
        ReLU

    Therefore one EdgeConv block has one main Linear weight matrix
    in its Edge MLP.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        k: int = 20,
        knn_chunk_size: int = 1024,
    ):
        super().__init__()

        if input_dim <= 0:
            raise ValueError(
                "input_dim must be positive."
            )

        if output_dim <= 0:
            raise ValueError(
                "output_dim must be positive."
            )

        if k <= 0:
            raise ValueError(
                "k must be positive."
            )

        if knn_chunk_size <= 0:
            raise ValueError(
                "knn_chunk_size must be positive."
            )

        self.input_dim = (
            input_dim
        )

        self.output_dim = (
            output_dim
        )

        self.k = k

        self.knn_chunk_size = (
            knn_chunk_size
        )

        # ---------------------------------------------------------
        # Shared Edge MLP
        #
        # Input:
        #     [f_i, f_j-f_i]
        #
        # Dimension:
        #     2 * input_dim
        #
        # Output:
        #     output_dim
        # ---------------------------------------------------------

        self.edge_linear = nn.Linear(
            2 * input_dim,
            output_dim,
        )

        self.activation = nn.ReLU()

    # -----------------------------------------------------------------

    def forward(
        self,
        features: torch.Tensor,
        knn_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        features:
            (B, N, input_dim)

        Returns
        -------
        point_features:
            (B, N, output_dim)
        """

        if features.ndim != 3:
            raise ValueError(
                f"Expected input shape (B, N, F), "
                f"found {tuple(features.shape)}."
            )

        if features.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected feature dimension "
                f"{self.input_dim}, found "
                f"{features.shape[-1]}."
            )

        # ---------------------------------------------------------
        # The graph feature space may differ from the edge-value
        # feature space.
        #
        # EdgeConv 1:
        #   k-NN          -> raw physical xyz
        #   edge values   -> normalized [x, y, z, velocity]
        #
        # EdgeConv 2 / 3:
        #   k-NN and edge values both use learned features.
        # ---------------------------------------------------------

        if knn_features is None:
            knn_features = features

        if knn_features.ndim != 3:
            raise ValueError(
                f"Expected knn_features shape (B, N, G), "
                f"found {tuple(knn_features.shape)}."
            )

        if knn_features.shape[:2] != features.shape[:2]:
            raise ValueError(
                "features and knn_features must have the same "
                "batch size and point count."
            )

        # =========================================================
        # 1. Dynamic k-NN
        # =========================================================

        neighbor_indices = (
            exact_knn_chunked(
                knn_features,
                k=self.k,
                chunk_size=(
                    self.knn_chunk_size
                ),
            )
        )

        # =========================================================
        # 2. [f_i, f_j-f_i]
        #
        # (B,N,k,F)
        #     +
        # (B,N,k,F)
        #
        # ->
        #
        # (B,N,k,2F)
        # =========================================================

        edge_features = (
            build_edge_features(
                features,
                neighbor_indices,
            )
        )

        # =========================================================
        # 3. Shared Edge MLP
        #
        # Same Linear weights for every edge.
        # =========================================================

        edge_features = (
            self.edge_linear(
                edge_features
            )
        )

        edge_features = (
            self.activation(
                edge_features
            )
        )

        # =========================================================
        # 4. Max pooling over k neighbors
        #
        # Before:
        #     (B,N,k,output_dim)
        #
        # After:
        #     (B,N,output_dim)
        # =========================================================

        point_features = (
            edge_features
            .max(
                dim=2,
            )
            .values
        )

        return point_features


# =====================================================================
# DGCNN regression model
# =====================================================================

class DGCNNRegressor(nn.Module):
    """
    Dynamic Graph CNN for point-wise CFD regression.

    Input:
        (B, N, 4)

        [x, y, z, velocity]

    Output:
        (B, N, 3)

        [HTC, wall_shear, pressure]

    Local features:
        EdgeConv1 -> 64
        EdgeConv2 -> 64
        EdgeConv3 -> 128

        concatenated local feature:
            256

    Global feature:
        global max pooling over all N points:
            256

    Final feature per point:
        local 256 + global 256
        =
        512
    """

    def __init__(
        self,
        input_dim: int = 4,
        k: int = 20,
        knn_chunk_size: int = 1024,
    ):
        super().__init__()

        if input_dim <= 0:
            raise ValueError(
                "input_dim must be positive."
            )

        self.input_dim = input_dim
        self.output_dim = 3

        self.k = k

        self.knn_chunk_size = (
            knn_chunk_size
        )

        # =========================================================
        # Dynamic EdgeConv blocks
        # =========================================================

        self.edgeconv1 = EdgeConv(
            input_dim=input_dim,
            output_dim=64,
            k=k,
            knn_chunk_size=(
                knn_chunk_size
            ),
        )

        self.edgeconv2 = EdgeConv(
            input_dim=64,
            output_dim=64,
            k=k,
            knn_chunk_size=(
                knn_chunk_size
            ),
        )

        self.edgeconv3 = EdgeConv(
            input_dim=64,
            output_dim=128,
            k=k,
            knn_chunk_size=(
                knn_chunk_size
            ),
        )

        # =========================================================
        # Regression head
        #
        # Local features:
        #     64 + 64 + 128
        #     =
        #     256
        #
        # Global feature:
        #     256
        #
        # Final:
        #     256 + 256
        #     =
        #     512
        #
        # Regression:
        #
        #     512 -> 256 -> 128 -> 3
        # =========================================================

        self.regression_head = (
            nn.Sequential(
                nn.Linear(
                    512,
                    256,
                ),
                nn.ReLU(),

                nn.Linear(
                    256,
                    128,
                ),
                nn.ReLU(),

                nn.Linear(
                    128,
                    3,
                ),
            )
        )

    # -----------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        first_knn_xyz: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x:
            Shape:
                (B, N, 4)

        Returns
        -------
        prediction:
            Shape:
                (B, N, 3)
        """

        if x.ndim != 3:
            raise ValueError(
                f"Expected input shape (B, N, {self.input_dim}), "
                f"found {tuple(x.shape)}."
            )

        if x.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected final input dimension "
                f"{self.input_dim}, found "
                f"{x.shape[-1]}."
            )

        if first_knn_xyz.ndim != 3:
            raise ValueError(
                f"Expected first_knn_xyz shape (B, N, 3), "
                f"found {tuple(first_knn_xyz.shape)}."
            )

        if first_knn_xyz.shape[-1] != 3:
            raise ValueError(
                f"Expected raw xyz dimension 3, found "
                f"{first_knn_xyz.shape[-1]}."
            )

        if first_knn_xyz.shape[:2] != x.shape[:2]:
            raise ValueError(
                "x and first_knn_xyz must have the same "
                "batch size and point count."
            )

        batch_size, num_points, _ = (
            x.shape
        )

        # =========================================================
        # EdgeConv 1
        #
        # (B,N,4)
        #     ↓
        # (B,N,64)
        # =========================================================

        x1 = self.edgeconv1(
            x,
            knn_features=first_knn_xyz,
        )

        # =========================================================
        # EdgeConv 2
        #
        # Dynamic k-NN is recomputed using x1.
        #
        # (B,N,64)
        #     ↓
        # (B,N,64)
        # =========================================================

        x2 = self.edgeconv2(
            x1
        )

        # =========================================================
        # EdgeConv 3
        #
        # Dynamic k-NN is recomputed using x2.
        #
        # (B,N,64)
        #     ↓
        # (B,N,128)
        # =========================================================

        x3 = self.edgeconv3(
            x2
        )

        # =========================================================
        # Local feature concatenation
        #
        # 64 + 64 + 128
        # =
        # 256
        #
        # Shape:
        #     (B,N,256)
        # =========================================================

        local_features = torch.cat(
            (
                x1,
                x2,
                x3,
            ),
            dim=-1,
        )

        # =========================================================
        # Global feature
        #
        # Take the maximum response of every learned feature
        # across all N points.
        #
        # (B,N,256)
        #      ↓ max over N
        # (B,1,256)
        #
        # This provides one learned descriptor of the entire
        # point cloud / face.
        # =========================================================

        global_features = (
            local_features
            .max(
                dim=1,
                keepdim=True,
            )
            .values
        )

        # =========================================================
        # Give the same global face descriptor to every point.
        #
        # (B,1,256)
        #      ↓
        # (B,N,256)
        #
        # expand() does not physically copy all values.
        # =========================================================

        global_features = (
            global_features
            .expand(
                batch_size,
                num_points,
                -1,
            )
        )

        # =========================================================
        # Local + global
        #
        # Each point now has:
        #
        #     local geometry information : 256
        #     whole-face information     : 256
        #
        # Total:
        #     512
        #
        # Shape:
        #     (B,N,512)
        # =========================================================

        final_features = torch.cat(
            (
                local_features,
                global_features,
            ),
            dim=-1,
        )

        # =========================================================
        # Point-wise regression
        #
        # (B,N,512)
        #     ↓
        # 512 -> 256 -> 128 -> 3
        #     ↓
        # (B,N,3)
        #
        # Final outputs:
        #     [...,0] = HTC
        #     [...,1] = wall_shear
        #     [...,2] = pressure
        # =========================================================

        prediction = (
            self.regression_head(
                final_features
            )
        )

        return prediction


# =====================================================================
# Utilities
# =====================================================================

def count_parameters(
    model: nn.Module,
) -> int:
    """Return number of trainable parameters."""

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


# =====================================================================
# Standalone self-test
# =====================================================================

def main():
    """
    Synthetic DGCNN self-test.

    Uses a small point cloud instead of the real N=7000.

    Tests:
        1. exact k-NN
        2. edge feature construction
        3. one EdgeConv
        4. full DGCNN output
        5. finite outputs
        6. final MSE backpropagation into EdgeConv 1
    """

    print("=" * 78)
    print("DGCNN MODEL SELF-TEST")
    print("=" * 78)

    torch.manual_seed(
        42
    )

    batch_size = 2
    num_points = 128
    input_dim = 4
    k = 20

    x = torch.randn(
        batch_size,
        num_points,
        input_dim,
        dtype=torch.float32,
        requires_grad=True,
    )

    # Raw xyz is used only to construct the first neighbor graph.
    raw_xyz = (
        x[
            ...,
            :3,
        ]
        .detach()
        .clone()
    )

    # =================================================================
    # Test 1: k-NN
    # =================================================================

    indices = exact_knn_chunked(
        raw_xyz,
        k=k,
        chunk_size=64,
    )

    print()
    print("[k-NN]")

    print(
        f"Input shape     : "
        f"{tuple(x.shape)}"
    )

    print(
        f"Neighbor shape  : "
        f"{tuple(indices.shape)}"
    )

    if indices.shape != (
        batch_size,
        num_points,
        k,
    ):
        raise RuntimeError(
            "k-NN shape test failed."
        )

    # =================================================================
    # Test 2: edge feature construction
    # =================================================================

    edges = build_edge_features(
        x,
        indices,
    )

    print()
    print("[EDGE FEATURES]")

    print(
        f"Expected shape  : "
        f"({batch_size}, "
        f"{num_points}, "
        f"{k}, "
        f"{2 * input_dim})"
    )

    print(
        f"Actual shape    : "
        f"{tuple(edges.shape)}"
    )

    if edges.shape != (
        batch_size,
        num_points,
        k,
        2 * input_dim,
    ):
        raise RuntimeError(
            "Edge-feature shape test failed."
        )

    # =================================================================
    # Test 3: one EdgeConv
    # =================================================================

    edgeconv = EdgeConv(
        input_dim=4,
        output_dim=64,
        k=k,
        knn_chunk_size=64,
    )

    edge_output = edgeconv(
        x,
        knn_features=raw_xyz,
    )

    print()
    print("[EDGE CONV]")

    print(
        f"Input shape     : "
        f"{tuple(x.shape)}"
    )

    print(
        f"Output shape    : "
        f"{tuple(edge_output.shape)}"
    )

    if edge_output.shape != (
        batch_size,
        num_points,
        64,
    ):
        raise RuntimeError(
            "EdgeConv shape test failed."
        )

    # =================================================================
    # Test 4: complete DGCNN
    # =================================================================

    model = DGCNNRegressor(
        input_dim=4,
        k=k,
        knn_chunk_size=64,
    )

    prediction = model(
        x,
        first_knn_xyz=raw_xyz,
    )

    print()
    print("[FULL MODEL]")
    print(model)

    print()

    print(
        f"Trainable parameters : "
        f"{count_parameters(model):,}"
    )

    print(
        f"Input shape          : "
        f"{tuple(x.shape)}"
    )

    print(
        f"Output shape         : "
        f"{tuple(prediction.shape)}"
    )

    if prediction.shape != (
        batch_size,
        num_points,
        3,
    ):
        raise RuntimeError(
            "Full-model output shape test failed."
        )

    if not torch.isfinite(
        prediction
    ).all():
        raise RuntimeError(
            "Model output contains NaN or Inf."
        )

    # =================================================================
    # Test 5: backpropagation
    # =================================================================

    target = torch.randn_like(
        prediction
    )

    loss = nn.MSELoss()(
        prediction,
        target,
    )

    loss.backward()

    first_weight_gradient = (
        model
        .edgeconv1
        .edge_linear
        .weight
        .grad
    )

    if first_weight_gradient is None:
        raise RuntimeError(
            "No gradient reached EdgeConv 1 weights."
        )

    if not torch.isfinite(
        first_weight_gradient
    ).all():
        raise RuntimeError(
            "EdgeConv gradient contains NaN or Inf."
        )

    regression_gradient = (
        model
        .regression_head[0]
        .weight
        .grad
    )

    if regression_gradient is None:
        raise RuntimeError(
            "No gradient reached regression head."
        )

    if not torch.isfinite(
        regression_gradient
    ).all():
        raise RuntimeError(
            "Regression-head gradient contains NaN or Inf."
        )

    print()
    print(
        f"Synthetic MSE loss   : "
        f"{loss.item():.6f}"
    )

    print()
    print("k-NN test            : PASS")
    print("Edge feature test    : PASS")
    print("EdgeConv test        : PASS")
    print("Global feature test  : PASS")
    print("Full model test      : PASS")
    print("Finite output test   : PASS")
    print("Backpropagation test : PASS")

    print()
    print("=" * 78)
    print("DGCNN MODEL SELF-TEST PASSED")
    print("=" * 78)


if __name__ == "__main__":
    main()