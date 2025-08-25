import numpy as np
from typing import Tuple, Union

ArrayLike = Union[float, int, np.ndarray]

class EmotionScaleAligner:
    """
    Unified scale alignment for emotion recognition systems.
    Reference space: valence, arousal in [-1, 1].
    """

    def __init__(self, strict: bool = False, dtype=np.float32):
        self.strict = strict
        self.dtype = dtype

    @staticmethod
    def _check_range(x, lo, hi, name, strict: bool):
        if strict:
            if np.any(x < lo) or np.any(x > hi):
                raise ValueError(f"{name} out of range [{lo}, {hi}].")
        return x

    @staticmethod
    def _clip(x, lo, hi):
        return np.clip(x, lo, hi)

    def _as_array(self, *xs):
        return [np.asarray(x, dtype=self.dtype) for x in xs]

    # ---------- FindingEmo ----------
    def findingemo_to_reference(self, v_fe: ArrayLike, a_fe: ArrayLike) -> Tuple[np.ndarray, np.ndarray]:
        """
        FE: valence in [-3, 3], arousal ('Intensity') in [0, 6] → reference [-1, 1].
        """
        v_fe, a_fe = self._as_array(v_fe, a_fe)
        v_fe = self._check_range(v_fe, -3.0, 3.0, "FE valence", self.strict)
        a_fe = self._check_range(a_fe,  0.0, 6.0, "FE arousal",  self.strict)

        # map, then clip to be safe
        v_ref = self._clip(v_fe / 3.0, -1.0, 1.0)
        a_ref = self._clip((a_fe / 3.0) - 1.0, -1.0, 1.0)
        return v_ref, a_ref

    def reference_to_findingemo(self, v_ref: ArrayLike, a_ref: ArrayLike) -> Tuple[np.ndarray, np.ndarray]:
        """
        reference [-1, 1] → FE: valence [-3, 3], arousal [0, 6].
        """
        v_ref, a_ref = self._as_array(v_ref, a_ref)
        v_ref = self._check_range(v_ref, -1.0, 1.0, "Ref valence", self.strict)
        a_ref = self._check_range(a_ref, -1.0, 1.0, "Ref arousal", self.strict)

        v_fe = self._clip(v_ref * 3.0, -3.0, 3.0)
        a_fe = self._clip((a_ref + 1.0) * 3.0, 0.0, 6.0)
        return v_fe, a_fe

    # ---------- DEAM (Static SAM 1..9) ----------
    def deam_static_to_reference(self, v_deam: ArrayLike, a_deam: ArrayLike) -> Tuple[np.ndarray, np.ndarray]:
        """
        DEAM static SAM [1, 9] → reference [-1, 1].
        """
        v_deam, a_deam = self._as_array(v_deam, a_deam)
        v_deam = self._check_range(v_deam, 1.0, 9.0, "DEAM valence", self.strict)
        a_deam = self._check_range(a_deam, 1.0, 9.0, "DEAM arousal", self.strict)

        v_ref = self._clip((v_deam - 5.0) / 4.0, -1.0, 1.0)
        a_ref = self._clip((a_deam - 5.0) / 4.0, -1.0, 1.0)
        return v_ref, a_ref

    def reference_to_deam_static(self, v_ref: ArrayLike, a_ref: ArrayLike) -> Tuple[np.ndarray, np.ndarray]:
        """
        reference [-1, 1] → DEAM static SAM [1, 9].
        """
        v_ref, a_ref = self._as_array(v_ref, a_ref)
        v_ref = self._check_range(v_ref, -1.0, 1.0, "Ref valence", self.strict)
        a_ref = self._check_range(a_ref, -1.0, 1.0, "Ref arousal", self.strict)

        v_deam = self._clip(5.0 + 4.0 * v_ref, 1.0, 9.0)
        a_deam = self._clip(5.0 + 4.0 * a_ref, 1.0, 9.0)
        return v_deam, a_deam

    # ---------- EmoNet ----------
    def emonet_to_reference(self, v_emonet: ArrayLike, a_emonet: ArrayLike) -> Tuple[np.ndarray, np.ndarray]:
        """
        EmoNet outputs are already in [-1, 1]. We just ensure range and dtype.
        """
        v_emonet, a_emonet = self._as_array(v_emonet, a_emonet)
        v_ref = self._clip(v_emonet, -1.0, 1.0)
        a_ref = self._clip(a_emonet, -1.0, 1.0)
        return v_ref, a_ref

    # ---------- Convenience methods for direct dataset conversions ----------
    def findingemo_to_deam_static(self, v_fe: ArrayLike, a_fe: ArrayLike) -> Tuple[np.ndarray, np.ndarray]:
        """
        Direct conversion: FindingEmo → DEAM static SAM [1, 9].
        """
        v_ref, a_ref = self.findingemo_to_reference(v_fe, a_fe)
        return self.reference_to_deam_static(v_ref, a_ref)

    def deam_static_to_findingemo(self, v_deam: ArrayLike, a_deam: ArrayLike) -> Tuple[np.ndarray, np.ndarray]:
        """
        Direct conversion: DEAM static SAM [1, 9] → FindingEmo.
        """
        v_ref, a_ref = self.deam_static_to_reference(v_deam, a_deam)
        return self.reference_to_findingemo(v_ref, a_ref)

    def emonet_to_findingemo(self, v_emonet: ArrayLike, a_emonet: ArrayLike) -> Tuple[np.ndarray, np.ndarray]:
        """
        Direct conversion: EmoNet → FindingEmo.
        """
        v_ref, a_ref = self.emonet_to_reference(v_emonet, a_emonet)
        return self.reference_to_findingemo(v_ref, a_ref)

    def emonet_to_deam_static(self, v_emonet: ArrayLike, a_emonet: ArrayLike) -> Tuple[np.ndarray, np.ndarray]:
        """
        Direct conversion: EmoNet → DEAM static SAM [1, 9].
        """
        v_ref, a_ref = self.emonet_to_reference(v_emonet, a_emonet)
        return self.reference_to_deam_static(v_ref, a_ref)
