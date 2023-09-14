import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, Field
from scipy import optimize
from scipy.stats import skewnorm


class SkewNormFitParameters(BaseModel):
    a: float = Field(
        description="scipy.stats.skewnorm takes a real number a as a skewness parameter. "
        "When a = 0, the distribution is identical to a normal distribution."
    )
    location: float
    scale: float
    pdf_scaling_constant: float
    covariance_matrix: npt.NDArray
    
    class Config:
        arbitrary_types_allowed = True


class ScanStats1D(BaseModel):
    skewness: float
    mean: float
    maximum_y_value: float
    sigma: float
    FWHM: float | None
    FWHM_x_coords: tuple[float, float] | list[float] | tuple[None, None]
    kurtosis: float
    skewnorm_fit_parameters: SkewNormFitParameters


def calculate_1D_scan_stats(x_array: npt.NDArray, y_array: npt.NDArray) -> ScanStats1D:
    """
    Calculates the skewness, mean ,maximum_y_value, sigma, full width at half maximum,
    kurtosis, and the skewnorm fit parameters (The skewnorm fit parameters
    are mainly for internal use).

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
        bounds = ([-1e-5, -np.inf, 0, 0], [np.inf, np.inf, np.inf, np.inf])
    else:
        bounds = ([-np.inf, -np.inf, 0, 0], [1e-5, np.inf, np.inf, np.inf])

    optimised_params, covariance_matrix = optimize.curve_fit(
        _skew_norm_fit_function,
        x_array,
        y_array,
        p0=[0, estimated_mean, estimated_sigma, estimated_pdf_scaling_constant],
        maxfev=4000,
        bounds=bounds,
    )
    a, location, scale, pdf_scaling_constant = optimised_params
    mean, variance, skewness, kurtosis = skewnorm.stats(
        a, loc=location, scale=scale, moments="mvsk"
    )

    x_tmp = np.linspace(min(x_array), max(x_array), 4096)
    y_tmp = pdf_scaling_constant * skewnorm.pdf(x_tmp, a, loc=location, scale=scale)
    FWHM_left, FWHM_right, FWHM = _full_width_at_half_maximum(x_tmp, y_tmp)

    return ScanStats1D(
        skewness=skewness,
        mean=mean,
        maximum_y_value=max(y_array),
        sigma=np.sqrt(variance),
        FWHM=FWHM,
        FWHM_x_coords=(FWHM_left, FWHM_right),
        kurtosis=kurtosis,
        skewnorm_fit_parameters=SkewNormFitParameters(
            a=a,
            location=location,
            scale=scale,
            pdf_scaling_constant=pdf_scaling_constant,
            covariance_matrix=covariance_matrix
        ),
    )


def _full_width_at_half_maximum(
    x_array: npt.NDArray, y_array: npt.NDArray
) -> tuple[float, float, float]:
    """Calculates the full width at half maximum (FWHM)

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
    if len(arg_y_limit) == 0:
        # The maximum has not been found
        return (None, None, None)
    arg_left = args_y[arg_y_limit][0]
    arg_right = args_y[arg_y_limit + 1][0]

    return (
        x_array[arg_left],
        x_array[arg_right],
        abs(x_array[arg_left] - x_array[arg_right]),
    )


def _skew_norm_fit_function(
    x: npt.NDArray,
    a: float,
    location: float,
    scale: float,
    pdf_scaling_constant: float,
) -> float:
    """
    Skewnorm fit function

    Parameters
    ----------
    x : npt.NDArray
        The x array
    a : float
        scipy.stats.skewnorm takes a real number `a` as a skewness parameter.
        When a = 0, the distribution is identical to a normal distribution.
    location : float
        The location of the distribution. When a=0, the location is the mean of the
        distribution
    scale : float
        The location of the distribution. When a=0, the location is the standard
        deviation of the distribution
    pdf_scaling_constant : float
        The probability density function scaling constant

    Returns
    -------
    float
        The PDF value times the pdf_scaling_constant
    """

    return pdf_scaling_constant * skewnorm.pdf(x, a, location, scale)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    a = 1
    x = np.linspace(-20, 100, 100)
    y = skewnorm.pdf(x, a, loc=50, scale=9.8)

    stats = calculate_1D_scan_stats(x, y)
    print("skewness:", stats.skewness)
    print("mean:", stats.mean)
    print("sigma:", stats.sigma)
    print("FWHM:", stats.FWHM)

    plt.plot(
        x,
        stats.skewnorm_fit_parameters.pdf_scaling_constant
        * skewnorm.pdf(
            x,
            stats.skewnorm_fit_parameters.a,
            stats.skewnorm_fit_parameters.location,
            stats.skewnorm_fit_parameters.scale,
        ),
        label="Fit",
    )
    plt.plot(x, y, linestyle="--", label="original data")
    plt.axvspan(xmin=stats.FWHM_x_coords[0], xmax=stats.FWHM_x_coords[1], alpha=0.2)

    plt.xlabel("x")
    plt.ylabel("PDF")
    plt.legend()
    plt.savefig("stats")
