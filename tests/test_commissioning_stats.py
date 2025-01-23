import numpy as np
import numpy.typing as npt
import pytest
from scipy.stats import skewnorm

from mx3_beamline_library.plans.commissioning.stats import Scan1DStats


@pytest.fixture()
def x_data() -> npt.NDArray:
    return np.linspace(-5, 5, 100)


@pytest.fixture()
def y_data() -> npt.NDArray:
    # Generate data with a skew normal distribution and Gaussian noise
    pdf_scaling_constant = 1
    a = 0
    location = 0
    scale = 1
    offset = 1

    x = np.linspace(-5, 5, 100)
    np.random.seed(0)
    y = (
        pdf_scaling_constant * skewnorm.pdf(x, a, location, scale)
        + offset
        + np.random.normal(0, 0.01, 100)
        + np.random.normal(0, 0.01, 100)
    )
    return y


@pytest.fixture()
def scan_stats(x_data, y_data) -> Scan1DStats:
    return Scan1DStats(x_data, y_data, flipped_gaussian=False)


def test_calculate_stats(scan_stats: Scan1DStats):
    result = scan_stats.calculate_stats()

    # test stats parameters
    assert result.mean == pytest.approx(-0.02, 0.1)
    assert result.skewness == pytest.approx(-0.04, 0.1)
    assert result.kurtosis == pytest.approx(0.01, 0.1)
    assert result.sigma == pytest.approx(0.99, 0.1)
    assert result.FWHM == pytest.approx(2.34, 0.1)
    assert result.FWHM_x_coords[0] == pytest.approx(-1.19, 0.1)
    assert result.FWHM_x_coords[1] == pytest.approx(1.15, 0.1)
    assert result.peak[0] == pytest.approx(-0.02, 0.2)
    assert result.peak[1] == pytest.approx(1.4, 0.1)

    # Test skew norm fit
    assert result.skewnorm_fit_parameters.a == pytest.approx(-0.58, 0.1)
    assert result.skewnorm_fit_parameters.location == pytest.approx(0.41, 0.1)
    assert result.skewnorm_fit_parameters.scale == pytest.approx(1.07, 0.1)
    assert result.skewnorm_fit_parameters.offset == pytest.approx(0.09, 0.1)
    assert result.skewnorm_fit_parameters.pdf_scaling_constant == pytest.approx(
        0.09, 0.1
    )


def test_full_width_at_half_maximum(scan_stats: Scan1DStats, x_data, y_data):
    left_coord, right_coord, abs_value = scan_stats._full_width_at_half_maximum(
        x_data, y_data
    )
    assert abs_value == pytest.approx(2.34, 0.2)
    assert left_coord == pytest.approx(-1.19, 0.2)
    assert right_coord == pytest.approx(1.15, 0.2)


def test_full_width_at_half_maximum_failure(scan_stats: Scan1DStats):
    x_data = np.linspace(-5, 5, 100)
    y_data = np.ones(100)
    left_coord, right_coord, abs_value = scan_stats._full_width_at_half_maximum(
        x_data, y_data
    )
    assert abs_value is None
    assert left_coord is None
    assert right_coord is None


def test_skewnorm_fit_function(scan_stats: Scan1DStats, x_data):
    x_data = np.array([-1, 0, 1])
    pdf_scaling_constant = 1
    a = 0
    location = 0
    scale = 1
    offset = 1

    result = scan_stats._skew_norm_fit_function(
        x_data, a, location, scale, offset, pdf_scaling_constant
    )

    assert result[0] == pytest.approx(1.24, 0.1)
    assert result[1] == pytest.approx(1.4, 0.1)
    assert result[2] == pytest.approx(1.24, 0.1)


def test_flipped_gaussian():

    pdf_scaling_constant = 1
    a = 0
    location = 0
    scale = 1
    offset = 1

    x = np.linspace(-5, 5, 100)
    np.random.seed(0)
    y = -1 * (
        pdf_scaling_constant * skewnorm.pdf(x, a, location, scale)
        + offset
        + np.random.normal(0, 0.01, 100)
        + np.random.normal(0, 0.01, 100)
    )

    stats_class = Scan1DStats(x, y, flipped_gaussian=True)
    result = stats_class.calculate_stats()

    assert result.peak[0] == pytest.approx(-0.02, 0.2)
    assert result.peak[1] == pytest.approx(-1.4, 0.1)
    # all other stats are the same as the non-flipped gaussian


def test_not_enough_data_points_failure():
    x = [1, 2, 3]
    y = [4, 5, 6]

    stats_class = Scan1DStats(x, y, flipped_gaussian=False)

    with pytest.warns(Warning):
        result = stats_class.calculate_stats()

    assert result is None


def test_calculate_stats_failure():
    x = [1, 2, 3, 4]
    y = [np.inf, 5, 6, 5]

    stats_class = Scan1DStats(x, y, flipped_gaussian=False)

    with pytest.warns(Warning):
        result = stats_class.calculate_stats()

    assert result is None
