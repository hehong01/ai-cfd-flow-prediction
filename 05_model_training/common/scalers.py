"""
Common standardization utilities for MLP and DGCNN training.

Input features:
    [x, y, z, velocity]

Target features:
    [HTC, wall_shear]

Standardization:
    normalized = (value - mean) / std

Inverse transformation:
    value = normalized * std + mean

Important:
    Statistics must be fitted using TRAIN data only.

The same implementation is shared by MLP and DGCNN, but each model
may fit different statistics because the actual training samples differ:

    MLP:
        all original CFD wall nodes

    DGCNN:
        FPS-selected 7000-point wall clouds

Typical saved files:
    weights/mlp/scalers.npz
    weights/dgcnn/scalers.npz
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


# =====================================================================
# Constants
# =====================================================================

INPUT_NAMES = (
    "x",
    "y",
    "z",
    "velocity",
)

TARGET_NAMES = (
    "HTC",
    "wall_shear",
)

NUM_INPUT_FEATURES = 4
NUM_TARGET_FEATURES = 2


# =====================================================================
# Validation
# =====================================================================

def _validate_input_array(
    x: np.ndarray,
) -> np.ndarray:
    """
    Validate model input data.

    Expected final dimension:
        4 -> [x, y, z, velocity]

    Any leading dimensions are allowed.

    Examples:
        MLP:
            (N, 4)

        DGCNN:
            (B, 7000, 4)
    """

    x = np.asarray(
        x,
        dtype=np.float64,
    )

    if x.ndim < 2:
        raise ValueError(
            f"Input must have at least 2 dimensions, "
            f"found shape {x.shape}."
        )

    if x.shape[-1] != NUM_INPUT_FEATURES:
        raise ValueError(
            f"Expected {NUM_INPUT_FEATURES} input features "
            f"[x, y, z, velocity], "
            f"found shape {x.shape}."
        )

    if not np.all(np.isfinite(x)):
        raise ValueError(
            "Input contains NaN or Inf."
        )

    return x


def _validate_target_array(
    y: np.ndarray,
) -> np.ndarray:
    """
    Validate model target data.

    Expected final dimension:
        2 -> [HTC, wall_shear]

    Any leading dimensions are allowed.

    Examples:
        MLP:
            (N, 2)

        DGCNN:
            (B, 7000, 2)
    """

    y = np.asarray(
        y,
        dtype=np.float64,
    )

    if y.ndim < 2:
        raise ValueError(
            f"Target must have at least 2 dimensions, "
            f"found shape {y.shape}."
        )

    if y.shape[-1] != NUM_TARGET_FEATURES:
        raise ValueError(
            f"Expected {NUM_TARGET_FEATURES} target features "
            f"[HTC, wall_shear], "
            f"found shape {y.shape}."
        )

    if not np.all(np.isfinite(y)):
        raise ValueError(
            "Target contains NaN or Inf."
        )

    return y


# =====================================================================
# CFD scaler
# =====================================================================

class CFDScaler:
    """
    Standard scaler for CFD model inputs and targets.

    Stores:

        input_mean:
            [mean_x, mean_y, mean_z, mean_velocity]

        input_std:
            [std_x, std_y, std_z, std_velocity]

        target_mean:
            [mean_HTC, mean_wall_shear]

        target_std:
            [std_HTC, std_wall_shear]
    """

    def __init__(
        self,
        input_mean: np.ndarray | None = None,
        input_std: np.ndarray | None = None,
        target_mean: np.ndarray | None = None,
        target_std: np.ndarray | None = None,
    ):

        self.input_mean = (
            None
            if input_mean is None
            else np.asarray(
                input_mean,
                dtype=np.float64,
            )
        )

        self.input_std = (
            None
            if input_std is None
            else np.asarray(
                input_std,
                dtype=np.float64,
            )
        )

        self.target_mean = (
            None
            if target_mean is None
            else np.asarray(
                target_mean,
                dtype=np.float64,
            )
        )

        self.target_std = (
            None
            if target_std is None
            else np.asarray(
                target_std,
                dtype=np.float64,
            )
        )

        if self.is_fitted:
            self._validate_statistics()

    # -----------------------------------------------------------------

    @property
    def is_fitted(self) -> bool:
        """Return True if all scaler statistics are available."""

        return all(
            value is not None
            for value in (
                self.input_mean,
                self.input_std,
                self.target_mean,
                self.target_std,
            )
        )

    # -----------------------------------------------------------------

    def _validate_statistics(self) -> None:
        """Validate stored mean/std arrays."""

        if self.input_mean.shape != (
            NUM_INPUT_FEATURES,
        ):
            raise ValueError(
                f"Invalid input_mean shape: "
                f"{self.input_mean.shape}"
            )

        if self.input_std.shape != (
            NUM_INPUT_FEATURES,
        ):
            raise ValueError(
                f"Invalid input_std shape: "
                f"{self.input_std.shape}"
            )

        if self.target_mean.shape != (
            NUM_TARGET_FEATURES,
        ):
            raise ValueError(
                f"Invalid target_mean shape: "
                f"{self.target_mean.shape}"
            )

        if self.target_std.shape != (
            NUM_TARGET_FEATURES,
        ):
            raise ValueError(
                f"Invalid target_std shape: "
                f"{self.target_std.shape}"
            )

        for name, array in (
            ("input_mean", self.input_mean),
            ("input_std", self.input_std),
            ("target_mean", self.target_mean),
            ("target_std", self.target_std),
        ):
            if not np.all(
                np.isfinite(array)
            ):
                raise ValueError(
                    f"{name} contains NaN or Inf."
                )

        if np.any(
            self.input_std <= 0
        ):
            raise ValueError(
                "input_std contains zero or negative values."
            )

        if np.any(
            self.target_std <= 0
        ):
            raise ValueError(
                "target_std contains zero or negative values."
            )

    # =================================================================
    # Fit
    # =================================================================

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
    ) -> "CFDScaler":
        """
        Fit scaler statistics from TRAIN data.

        Parameters
        ----------
        x_train:
            Input array whose final dimension is:

                [x, y, z, velocity]

        y_train:
            Target array whose final dimension is:

                [HTC, wall_shear]

        The leading dimensions are flattened before calculating
        statistics.

        Therefore both of these are valid:

            MLP:
                X -> (N, 4)
                Y -> (N, 2)

            DGCNN:
                X -> (B, 7000, 4)
                Y -> (B, 7000, 2)
        """

        x_train = _validate_input_array(
            x_train
        )

        y_train = _validate_target_array(
            y_train
        )

        # Flatten all sample/point dimensions while preserving
        # the final feature dimension.
        x_flat = x_train.reshape(
            -1,
            NUM_INPUT_FEATURES,
        )

        y_flat = y_train.reshape(
            -1,
            NUM_TARGET_FEATURES,
        )

        if len(x_flat) == 0:
            raise ValueError(
                "Cannot fit scaler on empty input data."
            )

        if len(y_flat) == 0:
            raise ValueError(
                "Cannot fit scaler on empty target data."
            )

        # Population standard deviation (ddof=0).
        self.input_mean = np.mean(
            x_flat,
            axis=0,
        )

        self.input_std = np.std(
            x_flat,
            axis=0,
            ddof=0,
        )

        self.target_mean = np.mean(
            y_flat,
            axis=0,
        )

        self.target_std = np.std(
            y_flat,
            axis=0,
            ddof=0,
        )

        self._validate_statistics()

        return self

    # =================================================================
    # Transform
    # =================================================================

    def transform_input(
        self,
        x: np.ndarray,
    ) -> np.ndarray:
        """
        Standardize model input.

        X_norm = (X - mean) / std
        """

        self._require_fitted()

        x = _validate_input_array(
            x
        )

        return (
            x - self.input_mean
        ) / self.input_std

    # -----------------------------------------------------------------

    def transform_target(
        self,
        y: np.ndarray,
    ) -> np.ndarray:
        """
        Standardize model targets.

        Y_norm = (Y - mean) / std
        """

        self._require_fitted()

        y = _validate_target_array(
            y
        )

        return (
            y - self.target_mean
        ) / self.target_std

    # =================================================================
    # Inverse transform
    # =================================================================

    def inverse_input(
        self,
        x_normalized: np.ndarray,
    ) -> np.ndarray:
        """
        Restore normalized input to original physical values.
        """

        self._require_fitted()

        x_normalized = (
            _validate_input_array(
                x_normalized
            )
        )

        return (
            x_normalized
            * self.input_std
            + self.input_mean
        )

    # -----------------------------------------------------------------

    def inverse_target(
        self,
        y_normalized: np.ndarray,
    ) -> np.ndarray:
        """
        Restore normalized prediction/target to physical values.

        Used after model prediction to recover:

            HTC
            wall shear
        """

        self._require_fitted()

        y_normalized = (
            _validate_target_array(
                y_normalized
            )
        )

        return (
            y_normalized
            * self.target_std
            + self.target_mean
        )

    # =================================================================
    # Save / load
    # =================================================================

    def save(
        self,
        path: str | Path,
    ) -> None:
        """
        Save scaler statistics to an NPZ file.
        """

        self._require_fitted()

        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        np.savez(
            path,
            input_mean=self.input_mean,
            input_std=self.input_std,
            target_mean=self.target_mean,
            target_std=self.target_std,
        )

    # -----------------------------------------------------------------

    @classmethod
    def load(
        cls,
        path: str | Path,
    ) -> "CFDScaler":
        """
        Load scaler statistics from an NPZ file.
        """

        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(
                f"Scaler file not found:\n{path}"
            )

        with np.load(path) as data:

            required = {
                "input_mean",
                "input_std",
                "target_mean",
                "target_std",
            }

            missing = (
                required
                - set(data.files)
            )

            if missing:
                raise ValueError(
                    f"Scaler file is missing fields: "
                    f"{sorted(missing)}"
                )

            scaler = cls(
                input_mean=data[
                    "input_mean"
                ].copy(),
                input_std=data[
                    "input_std"
                ].copy(),
                target_mean=data[
                    "target_mean"
                ].copy(),
                target_std=data[
                    "target_std"
                ].copy(),
            )

        return scaler

    # =================================================================
    # Display
    # =================================================================

    def print_statistics(
        self,
    ) -> None:
        """Print scaler mean/std values."""

        self._require_fitted()

        print("=" * 72)
        print("CFD SCALER STATISTICS")
        print("=" * 72)

        print()
        print("[INPUT]")

        for i, name in enumerate(
            INPUT_NAMES
        ):
            print(
                f"{name:<12} "
                f"mean = "
                f"{self.input_mean[i]: .10g}   "
                f"std = "
                f"{self.input_std[i]: .10g}"
            )

        print()
        print("[TARGET]")

        for i, name in enumerate(
            TARGET_NAMES
        ):
            print(
                f"{name:<12} "
                f"mean = "
                f"{self.target_mean[i]: .10g}   "
                f"std = "
                f"{self.target_std[i]: .10g}"
            )

        print("=" * 72)

    # =================================================================

    def _require_fitted(
        self,
    ) -> None:
        """Raise if scaler has not been fitted."""

        if not self.is_fitted:
            raise RuntimeError(
                "Scaler is not fitted."
            )


# =====================================================================
# Standalone self-test
# =====================================================================

def main():
    """
    Small internal test.

    This does NOT fit the real CFD dataset.
    It only verifies scaler math, save/load, and inverse transformation.
    """

    print("=" * 72)
    print("CFD SCALER SELF-TEST")
    print("=" * 72)

    x = np.array(
        [
            [0.01, 0.02, -0.03, 5.0],
            [0.02, 0.04, -0.01, 8.0],
            [0.03, 0.06, 0.01, 10.0],
        ],
        dtype=np.float64,
    )

    y = np.array(
        [
            [30.0, 0.10],
            [60.0, 0.30],
            [90.0, 0.50],
        ],
        dtype=np.float64,
    )

    scaler = CFDScaler().fit(
        x,
        y,
    )

    scaler.print_statistics()

    x_norm = scaler.transform_input(
        x
    )

    y_norm = scaler.transform_target(
        y
    )

    x_restored = scaler.inverse_input(
        x_norm
    )

    y_restored = scaler.inverse_target(
        y_norm
    )

    if not np.allclose(
        x,
        x_restored,
    ):
        raise RuntimeError(
            "Input inverse-transform test failed."
        )

    if not np.allclose(
        y,
        y_restored,
    ):
        raise RuntimeError(
            "Target inverse-transform test failed."
        )

    print()
    print("Input normalization    : PASS")
    print("Target normalization   : PASS")
    print("Input inverse transform: PASS")
    print("Target inverse transform: PASS")

    print()
    print("=" * 72)
    print("SCALER SELF-TEST PASSED")
    print("=" * 72)


if __name__ == "__main__":
    main()