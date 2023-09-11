import numpy as np
import numpy.typing as npt
from pydantic import BaseModel
from scipy import optimize
from scipy.stats import skewnorm


class ScanStats1D(BaseModel):
    skewness: float
    mean: float
    maximum_y_value: float
    sigma: float
    FWHM: float
    FWHM_x_coords: tuple[float, float] | list[float]
    pdf_scaling_constant: float


def calculate_1D_scan_stats(x_array: npt.NDArray, y_array: npt.NDArray) -> ScanStats1D:
    """Calculates the skewness, mean ,maximum_y_value, sigma,
    full width at half maximum, and pdf scaling constant
    based on x_array and y_array

    Parameters
    ----------
    x_array : npt.NDArray
        The x array, for example an array containing motor positions
    y_array : npt.NDArray
        The y array, for example the intensity array

    Returns
    -------
    ScanStats1D
        A ScanStats1D pydantic model
    """

    estimated_mean = sum(x_array * y_array) / sum(y_array)
    estimated_mode = x_array[np.argmax(y_array)]
    estimated_sigma = np.sqrt(
        sum(y_array * (x_array - estimated_mean) ** 2) / sum(y_array)
    )
    estimated_pdf_scaling_constant = max(y_array) / max(y_array / (sum(y_array)))
    # if mean>mode, the distribution is positively skewed
    estimated_skewness_sign = estimated_mean - estimated_mode

    if estimated_skewness_sign >= 0:
        bounds = ([-1e-5, 0, 0, 0], [np.inf, np.inf, np.inf, np.inf])
    else:
        bounds = ([-np.inf, 0, 0, 0], [1e-5, np.inf, np.inf, np.inf])

    optimised_params, covariance_matrix = optimize.curve_fit(
        _skew_norm_fit_function,
        x_array,
        y_array,
        p0=[0, estimated_mean, estimated_sigma, estimated_pdf_scaling_constant],
        maxfev=4000,
        bounds=bounds,
    )
    skewness, mean, sigma, pdf_scaling_constant = optimised_params

    x_new = np.linspace(min(x_array), max(x_array), 4096)
    y_new = pdf_scaling_constant * skewnorm.pdf(x_new, skewness, mean, sigma)

    FWHM_left, FWHM_right, FWHM = _full_width_at_half_maximum(x_new, y_new)

    return ScanStats1D(
        skewness=skewness,
        mean=mean,
        maximum_y_value=max(y_array),
        sigma=sigma,
        FWHM=FWHM,
        FWHM_x_coords=(FWHM_left, FWHM_right),
        pdf_scaling_constant=pdf_scaling_constant,
    )


def _full_width_at_half_maximum(
    x_array: npt.NDArray, y_array: npt.NDArray
) -> tuple[float, float, float]:
    """Calculated the full width at half maximum (FWHM)

    Parameters
    ----------
    x_array : npt.NDArray
        The x array, for example an array containing motor positions
    y_array : npt.NDArray
        The y array, for example the intensity array

    Returns
    -------
    tuple[float, float, float]
        The left x coordinated of the FWHM, the right x coordinate of the FWHM
        and the FWHM
    """
    args_y = np.where(y_array < np.max(y_array) / 2)[0]

    arg_y_limit = np.where(np.diff(args_y) > 1)[0]

    arg_left = args_y[arg_y_limit][0]
    arg_right = args_y[arg_y_limit + 1][0]

    return (
        x_array[arg_left],
        x_array[arg_right],
        abs(x_array[arg_left] - x_array[arg_right]),
    )


def _skew_norm_fit_function(
    x: npt.NDArray,
    skewness: float,
    mean: float,
    sigma: float,
    pdf_scaling_constant: float,
) -> float:
    """_summary_

    Parameters
    ----------
    x : npt.NDArray
        The x array
    skewness : float
        The skewness
    mean : float
        The mean
    sigma : float
        Sigma
    pdf_scaling_constant : float
        the probability density function scaling constant

    Returns
    -------
    float
        The pdf value times the pdf_scaling_constant
    """

    return pdf_scaling_constant * skewnorm.pdf(x, skewness, mean, sigma)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    a = 3
    x = np.linspace(-20, 100, 100)
    y = skewnorm.pdf(x, a, loc=50, scale=9.8)

    stats = calculate_1D_scan_stats(x, y)
    plt.plot(
        x,
        stats.pdf_scaling_constant
        * skewnorm.pdf(x, stats.skewness, stats.mean, stats.sigma),
    )
    plt.plot(x, y, linestyle="--")
    plt.axvline(stats.FWHM_x_coords[0])
    plt.axvline(stats.FWHM_x_coords[1])

    plt.xlabel("x")
    plt.ylabel("PDF")
    plt.savefig("stats")
