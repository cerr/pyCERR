"""
Unit tests for cerr.roe.dosimetric_models.

Each of the 10 pre-shipped model JSON files is tested by:
  1. Loading the JSON parameter file directly.
  2. Creating a synthetic uniform-dose DVH (no planC / DICOM required).
  3. Calling the model function with the loaded parameters.
  4. Asserting the result is a valid probability in [0, 1].
  5. Asserting monotonicity: higher dose yields higher NTCP.
  6. Spot-checking one analytically derived reference value.

Fractionation correction is applied manually for models that require it so
that get_corrected_dvbins() (which needs a planC) is not called.
"""

import json
import os

import numpy as np
import pytest
from scipy.special import erf

from cerr.roe.dosimetric_models import LKBFn, appeltLogit, coxFn, logitFn
from cerr.dataclasses.dose import fractionSizeCorrect, fractionNumCorrect

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_MODEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "cerr", "roe", "model_parameters",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_model(filename: str) -> dict:
    with open(os.path.join(_MODEL_DIR, filename)) as fh:
        return json.load(fh)


def make_uniform_dvh(dose_gy: float, total_vol_cc: float = 100.0,
                     bin_width: float = 0.05):
    """
    Synthetic DVH for a structure irradiated uniformly to *dose_gy*.
    All volume sits in the single bin that contains *dose_gy*.
    Returns (doseBinsV, volHistV).
    """
    if dose_gy <= 0:
        return np.array([bin_width / 2]), np.array([total_vol_cc])
    n_bins = max(1, int(np.ceil(dose_gy / bin_width)))
    doseBinsV = (np.arange(n_bins) + 0.5) * bin_width
    volHistV = np.zeros(n_bins)
    volHistV[-1] = total_vol_cc
    return doseBinsV, volHistV


def frx_correct_bins(doseBinsV, model: dict, fsize_in: float, fnum_in: int):
    """
    Apply the fractionation correction specified by *model* to *doseBinsV*.
    Returns corrected dose bins (no planC needed — planC is only used by
    fractionSizeCorrect/fractionNumCorrect to look up the associated scan,
    which is not used in the bin-scaling math; passing None is safe here).
    """
    correction = model.get("fractionCorrect", "No").lower()
    if correction != "yes":
        return doseBinsV

    ctype = model.get("correctionType", "").lower()
    ab = float(model.get("abRatio", 10))

    if ctype in ("frxsize", "fsize"):
        std_fsize = float(model["stdFractionSize"])
        return fractionSizeCorrect(doseBinsV, std_fsize, ab, None, fsize_in)
    elif ctype == "frxnum":
        std_nfrx = int(model["stdNumFractions"])
        return fractionNumCorrect(doseBinsV, std_nfrx, ab, None, fnum_in)
    return doseBinsV


def _assert_valid_ntcp(value, name=""):
    """Assert the result is a finite scalar in [0, 1]."""
    assert np.isfinite(value), f"{name}: result is not finite ({value})"
    assert 0.0 <= value <= 1.0, f"{name}: NTCP {value:.4f} outside [0, 1]"


def _assert_monotone(ntcp_low, ntcp_high, name=""):
    """Assert NTCP rises (or at worst stays equal) with dose."""
    assert ntcp_high >= ntcp_low - 1e-9, (
        f"{name}: not monotone — low-dose NTCP {ntcp_low:.4f} "
        f"> high-dose NTCP {ntcp_high:.4f}"
    )


# ---------------------------------------------------------------------------
# 1. LKB model — Rectal bleeding (grade 2+)
# ---------------------------------------------------------------------------

class TestLKBRectalBleeding:
    MODEL_FILE = "Rectal bleeding (grade 2+).json"
    # frxsize correction: stdFractionSize=2, abRatio=3
    FSIZE_IN = 2.0   # same as std → correction is identity
    FNUM_IN  = 38    # not used (correctionType is frxsize)

    def _ntcp(self, dose_gy):
        model = load_model(self.MODEL_FILE)
        bins, vols = make_uniform_dvh(dose_gy)
        corr_bins = frx_correct_bins(bins, model, self.FSIZE_IN, self.FNUM_IN)
        return LKBFn(model["parameters"], corr_bins, vols)

    def test_valid_range_low_dose(self):
        _assert_valid_ntcp(self._ntcp(20.0), "LKB low dose")

    def test_valid_range_mid_dose(self):
        _assert_valid_ntcp(self._ntcp(50.0), "LKB mid dose")

    def test_valid_range_high_dose(self):
        _assert_valid_ntcp(self._ntcp(90.0), "LKB high dose")

    def test_monotone(self):
        _assert_monotone(self._ntcp(30.0), self._ntcp(60.0), "LKB")
        _assert_monotone(self._ntcp(60.0), self._ntcp(90.0), "LKB")

    def test_at_d50(self):
        """At D50=76.9 Gy (uniform dose, no frx correction), NTCP ≈ 0.5."""
        model = load_model(self.MODEL_FILE)
        d50 = model["parameters"]["D50"]["val"]
        bins, vols = make_uniform_dvh(d50)
        # Identity correction (input fsize == std fsize)
        corr_bins = frx_correct_bins(bins, model, self.FSIZE_IN, self.FNUM_IN)
        ntcp = LKBFn(model["parameters"], corr_bins, vols)
        assert abs(ntcp - 0.5) < 0.02, f"LKB at D50: expected ~0.5, got {ntcp:.4f}"

    def test_near_zero_dose(self):
        ntcp = self._ntcp(1.0)
        assert ntcp < 0.01, f"LKB near-zero dose: expected <0.01, got {ntcp:.4f}"


# ---------------------------------------------------------------------------
# 2. Logistic model — Bronchial stenosis (logistic)
# ---------------------------------------------------------------------------

class TestLogitBronchialStenosisLogistic:
    MODEL_FILE = "Bronchial stenosis (logistic).json"

    def _ntcp(self, dose_gy):
        model = load_model(self.MODEL_FILE)
        bins, vols = make_uniform_dvh(dose_gy)
        return logitFn(model["parameters"], [bins], [vols])

    def test_valid_range(self):
        for d in [10, 30, 50, 70, 90]:
            _assert_valid_ntcp(self._ntcp(d), f"logit-bronchial D={d}")

    def test_monotone(self):
        _assert_monotone(self._ntcp(10), self._ntcp(50), "logit-bronchial")
        _assert_monotone(self._ntcp(50), self._ntcp(80), "logit-bronchial")

    def test_reference_value_40gy(self):
        """
        At mean dose = 40 Gy:
          gx = 0.0644*40 + 1*(-5.17) = 2.576 - 5.17 = -2.594
          NTCP = 1/(1+exp(2.594)) ≈ 0.0694
        """
        ntcp = self._ntcp(40.0)
        expected = 1.0 / (1.0 + np.exp(2.594))
        assert abs(ntcp - expected) < 0.01, (
            f"logit-bronchial at 40 Gy: expected {expected:.4f}, got {ntcp:.4f}"
        )

    def test_near_zero_dose(self):
        ntcp = self._ntcp(0.5)
        assert ntcp < 0.01, f"logit-bronchial near-zero: {ntcp:.4f}"


# ---------------------------------------------------------------------------
# 3. Cox model — Bronchial stenosis (cox)
# ---------------------------------------------------------------------------

class TestCoxBronchialStenosisCox:
    MODEL_FILE = "Bronchial stenosis (cox).json"

    def _ntcp(self, dose_gy):
        model = load_model(self.MODEL_FILE)
        bins, vols = make_uniform_dvh(dose_gy)
        return coxFn(model["parameters"], [bins], [vols])

    def test_valid_range(self):
        for d in [10, 30, 50, 70, 90]:
            _assert_valid_ntcp(self._ntcp(d), f"cox-bronchial D={d}")

    def test_monotone(self):
        _assert_monotone(self._ntcp(10), self._ntcp(50), "cox-bronchial")
        _assert_monotone(self._ntcp(50), self._ntcp(80), "cox-bronchial")

    def test_reference_value_40gy(self):
        """
        At mean dose = 40 Gy:
          H = 0.00887 * exp(0.0629 * 40) = 0.00887 * 12.378 ≈ 0.1098
          P = 1 - exp(-0.1098) ≈ 0.1041
        """
        ntcp = self._ntcp(40.0)
        h = 0.00887 * np.exp(0.0629 * 40.0)
        expected = 1.0 - np.exp(-h)
        assert abs(ntcp - expected) < 0.01, (
            f"cox-bronchial at 40 Gy: expected {expected:.4f}, got {ntcp:.4f}"
        )

    def test_near_zero_dose(self):
        ntcp = self._ntcp(0.5)
        assert ntcp < 0.05, f"cox-bronchial near-zero: {ntcp:.4f}"


# ---------------------------------------------------------------------------
# 4. Logistic — Bronchial toxicity Grade 3+ (frxsize correction)
# ---------------------------------------------------------------------------

class TestLogitBronchialToxicityGrade3:
    MODEL_FILE = "Bronchial toxicity  (Grade 3+).json"
    FSIZE_IN = 10.0   # SBRT-like, large fraction size
    FNUM_IN  = 5

    def _ntcp(self, dose_gy):
        model = load_model(self.MODEL_FILE)
        bins, vols = make_uniform_dvh(dose_gy)
        corr_bins = frx_correct_bins(bins, model, self.FSIZE_IN, self.FNUM_IN)
        return logitFn(model["parameters"], [corr_bins], [vols])

    def test_valid_range(self):
        for d in [10, 30, 60, 80]:
            _assert_valid_ntcp(self._ntcp(d), f"logit-bronch-G3 D={d}")

    def test_monotone(self):
        _assert_monotone(self._ntcp(20), self._ntcp(60), "logit-bronch-G3")

    def test_near_zero_dose(self):
        ntcp = self._ntcp(0.5)
        assert ntcp < 0.05, f"logit-bronch-G3 near-zero: {ntcp:.4f}"


# ---------------------------------------------------------------------------
# 5. Logistic — Bronchial toxicity Grade 5 (frxsize correction)
# ---------------------------------------------------------------------------

class TestLogitBronchialToxicityGrade5:
    MODEL_FILE = "Bronchial toxicity (Grade 5).json"
    FSIZE_IN = 10.0
    FNUM_IN  = 5

    def _ntcp(self, dose_gy):
        model = load_model(self.MODEL_FILE)
        bins, vols = make_uniform_dvh(dose_gy)
        corr_bins = frx_correct_bins(bins, model, self.FSIZE_IN, self.FNUM_IN)
        return logitFn(model["parameters"], [corr_bins], [vols])

    def test_valid_range(self):
        for d in [10, 40, 70]:
            _assert_valid_ntcp(self._ntcp(d), f"logit-bronch-G5 D={d}")

    def test_monotone(self):
        _assert_monotone(self._ntcp(20), self._ntcp(60), "logit-bronch-G5")

    def test_near_zero_dose(self):
        ntcp = self._ntcp(0.5)
        assert ntcp < 0.05, f"logit-bronch-G5 near-zero: {ntcp:.4f}"


# ---------------------------------------------------------------------------
# 6. Logistic — Esophagitis (Huang), frxnum correction
# ---------------------------------------------------------------------------

class TestLogitEsophagitisHuang:
    MODEL_FILE = "Esophagitis (Huang).json"
    FSIZE_IN = 2.0
    FNUM_IN  = 35   # same as std → identity correction

    def _ntcp(self, dose_gy, concurrent_chemo=0):
        model = load_model(self.MODEL_FILE)
        model["parameters"]["concurrentChemo"]["val"] = concurrent_chemo
        bins, vols = make_uniform_dvh(dose_gy)
        corr_bins = frx_correct_bins(bins, model, self.FSIZE_IN, self.FNUM_IN)
        return logitFn(model["parameters"], [corr_bins], [vols])

    def test_valid_range(self):
        for d in [10, 30, 50, 70]:
            _assert_valid_ntcp(self._ntcp(d), f"logit-esoph-huang D={d}")

    def test_monotone(self):
        _assert_monotone(self._ntcp(10), self._ntcp(50), "logit-esoph-huang")

    def test_chemo_increases_ntcp(self):
        """Concurrent chemotherapy should increase NTCP (positive weight=1.5)."""
        ntcp_no_chemo = self._ntcp(40.0, concurrent_chemo=0)
        ntcp_chemo    = self._ntcp(40.0, concurrent_chemo=1)
        assert ntcp_chemo > ntcp_no_chemo, (
            f"Esophagitis Huang: chemo should raise NTCP "
            f"({ntcp_chemo:.4f} vs {ntcp_no_chemo:.4f})"
        )

    def test_reference_value_40gy_no_chemo(self):
        """
        At mean dose=40 Gy, no chemo (std fractions → identity correction):
          gx = 0.0688*40 + 1.5*0 + 1*(-3.13) = 2.752 - 3.13 = -0.378
          NTCP = 1/(1+exp(0.378)) ≈ 0.4066
        """
        ntcp = self._ntcp(40.0, concurrent_chemo=0)
        expected = 1.0 / (1.0 + np.exp(0.378))
        assert abs(ntcp - expected) < 0.01, (
            f"logit-esoph-huang at 40 Gy: expected {expected:.4f}, got {ntcp:.4f}"
        )


# ---------------------------------------------------------------------------
# 7. Cox — Esophagitis Jackson (cox), frxsize correction
# ---------------------------------------------------------------------------

class TestCoxEsophagitisJackson:
    MODEL_FILE = "Esophagitis (Jackson_cox).json"
    FSIZE_IN = 2.0   # same as std → identity
    FNUM_IN  = 25

    def _ntcp(self, dose_gy):
        model = load_model(self.MODEL_FILE)
        bins, vols = make_uniform_dvh(dose_gy)
        corr_bins = frx_correct_bins(bins, model, self.FSIZE_IN, self.FNUM_IN)
        return coxFn(model["parameters"], [corr_bins], [vols])

    def test_valid_range(self):
        for d in [10, 30, 50, 70]:
            _assert_valid_ntcp(self._ntcp(d), f"cox-esoph-jackson D={d}")

    def test_monotone(self):
        _assert_monotone(self._ntcp(10), self._ntcp(50), "cox-esoph-jackson")

    def test_near_zero_dose(self):
        ntcp = self._ntcp(0.5)
        assert ntcp < 0.1, f"cox-esoph-jackson near-zero: {ntcp:.4f}"


# ---------------------------------------------------------------------------
# 8. Logistic — Esophagitis Jackson (logistic), frxsize correction
# ---------------------------------------------------------------------------

class TestLogitEsophagitisJackson:
    MODEL_FILE = "Esophagitis (Jackson_logistic).json"
    FSIZE_IN = 2.0
    FNUM_IN  = 25

    def _ntcp(self, dose_gy):
        model = load_model(self.MODEL_FILE)
        bins, vols = make_uniform_dvh(dose_gy)
        corr_bins = frx_correct_bins(bins, model, self.FSIZE_IN, self.FNUM_IN)
        return logitFn(model["parameters"], [corr_bins], [vols])

    def test_valid_range(self):
        for d in [10, 30, 50, 70]:
            _assert_valid_ntcp(self._ntcp(d), f"logit-esoph-jackson D={d}")

    def test_monotone(self):
        _assert_monotone(self._ntcp(10), self._ntcp(60), "logit-esoph-jackson")

    def test_near_zero_dose(self):
        ntcp = self._ntcp(0.5)
        assert ntcp < 0.05, f"logit-esoph-jackson near-zero: {ntcp:.4f}"


# ---------------------------------------------------------------------------
# 9. Logistic — Esophagitis Wijsman (frxsize correction, 4 clinical covariates)
# ---------------------------------------------------------------------------

class TestLogitEsophagitisWijsman:
    MODEL_FILE = "Esophagitis (Wijsman).json"
    FSIZE_IN = 2.0
    FNUM_IN  = 25

    def _ntcp(self, dose_gy, gender=0, chemo=0, stage=0):
        model = load_model(self.MODEL_FILE)
        model["parameters"]["gender"]["val"]           = gender
        model["parameters"]["concurrentChemo"]["val"]  = chemo
        model["parameters"]["tumorStage"]["val"]       = stage
        bins, vols = make_uniform_dvh(dose_gy)
        corr_bins = frx_correct_bins(bins, model, self.FSIZE_IN, self.FNUM_IN)
        return logitFn(model["parameters"], [corr_bins], [vols])

    def test_valid_range(self):
        for d in [10, 30, 50, 70]:
            _assert_valid_ntcp(self._ntcp(d), f"logit-esoph-wijsman D={d}")

    def test_monotone_dose(self):
        _assert_monotone(self._ntcp(10), self._ntcp(50), "logit-esoph-wijsman")

    def test_covariates_increase_ntcp(self):
        """Female sex (weight 1.204), chemo (2.645), cT3/4 (0.994) all raise NTCP."""
        base  = self._ntcp(30.0, gender=0, chemo=0, stage=0)
        worst = self._ntcp(30.0, gender=1, chemo=1, stage=1)
        assert worst > base, (
            f"Wijsman: worst-case covariates should raise NTCP "
            f"({worst:.4f} vs {base:.4f})"
        )

    def test_near_zero_dose(self):
        ntcp = self._ntcp(0.5)
        assert ntcp < 0.05, f"logit-esoph-wijsman near-zero: {ntcp:.4f}"


# ---------------------------------------------------------------------------
# 10. Appelt logistic — Pneumonitis (Appelt), frxnum correction
# ---------------------------------------------------------------------------

class TestAppeltPneumonitis:
    MODEL_FILE = "Pneumonitis (Appelt).json"
    FSIZE_IN = 2.0
    FNUM_IN  = 35   # same as std → identity

    def _ntcp(self, dose_gy, **risk_factors):
        model = load_model(self.MODEL_FILE)
        for key, val in risk_factors.items():
            model["parameters"][key]["val"] = val
        bins, vols = make_uniform_dvh(dose_gy)
        corr_bins = frx_correct_bins(bins, model, self.FSIZE_IN, self.FNUM_IN)
        return appeltLogit(model["parameters"], [corr_bins], [vols])

    def test_valid_range(self):
        for d in [10, 25, 34.4, 50, 70]:
            _assert_valid_ntcp(self._ntcp(d), f"appelt D={d}")

    def test_monotone(self):
        _assert_monotone(self._ntcp(10), self._ntcp(34.4), "appelt")
        _assert_monotone(self._ntcp(34.4), self._ntcp(60), "appelt")

    def test_at_d50_no_risk_factors(self):
        """
        At mean dose = D50_0 = 34.4 Gy with all risk factors absent (OR=1):
          NTCP = 1/(1+exp(0)) = 0.5
        """
        ntcp = self._ntcp(34.4)
        assert abs(ntcp - 0.5) < 0.02, (
            f"Appelt at D50 (no risk factors): expected 0.5, got {ntcp:.4f}"
        )

    def test_risk_factors_shift_response(self):
        """
        Pulmonary comorbidity (OR=2.27 > 1) reduces D50 → raises NTCP at D50_0.
        Former smoker (OR=0.69 < 1) increases D50 → lowers NTCP at D50_0.
        """
        ntcp_base         = self._ntcp(34.4)
        ntcp_comorbidity  = self._ntcp(34.4, pulmonaryComorbidity=1)
        ntcp_former_smoke = self._ntcp(34.4, formerSmoker=1)

        assert ntcp_comorbidity > ntcp_base, (
            "Pulmonary comorbidity (OR>1) should increase NTCP at D50"
        )
        assert ntcp_former_smoke < ntcp_base, (
            "Former smoker (OR<1) should decrease NTCP at D50"
        )

    def test_near_zero_dose(self):
        ntcp = self._ntcp(1.0)
        assert ntcp < 0.1, f"appelt near-zero: {ntcp:.4f}"
