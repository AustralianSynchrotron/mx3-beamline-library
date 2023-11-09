from typing import Union
from warnings import warn

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
    peak: tuple[float, float]
    sigma: float
    FWHM: float | None
    FWHM_x_coords: tuple[float, float] | list[float] | tuple[None, None]
    kurtosis: float
    skewnorm_fit_parameters: SkewNormFitParameters


class Scan1DStats:
    def __init__(
        self,
        x_array: Union[npt.NDArray, list],
        y_array: Union[npt.NDArray, list],
        flipped_gaussian=False,
    ) -> None:
        """
        Parameters
        ----------
        x_array : Union[npt.NDArray, list]
            The x array, for example an array containing motor positions
        y_array : Union[npt.NDArray, list]
            The y array, for example the intensity array
        flipped_gaussian: bool, optional
            Determines if the function that we are fitting is a Gaussian distribution
            flipped upside-down, by default False

        Returns
        -------
        None
        """
        self.x_array = np.array(x_array)
        self.y_array = np.array(y_array)
        self.flipped_gaussian = flipped_gaussian

        if flipped_gaussian:
            self.y_array = np.array(y_array) * -1

    def calculate_stats(self) -> ScanStats1D:
        """
        Calculates the skewness, mean ,maximum_y_value, sigma, full width at half maximum,
        kurtosis, and the skewnorm fit parameters (The skewnorm fit parameters
        are mainly for internal use).


        Returns
        -------
        ScanStats1D
            A ScanStats1D pydantic model
        """
        estimated_mean = sum(self.x_array * self.y_array) / sum(self.y_array)
        estimated_mode = self.x_array[np.argmax(self.y_array)]
        estimated_sigma = np.sqrt(
            sum(self.y_array * (self.x_array - estimated_mean) ** 2) / sum(self.y_array)
        )
        estimated_pdf_scaling_constant = max(self.y_array) / max(
            self.y_array / (sum(self.y_array))
        )
        # if mean>mode, the distribution is positively skewed
        estimated_skewness_sign = estimated_mean - estimated_mode

        if estimated_skewness_sign >= 0:
            bounds = ([-1e-5, -np.inf, 0, 0], [np.inf, np.inf, np.inf, np.inf])
        else:
            bounds = ([-np.inf, -np.inf, 0, 0], [1e-5, np.inf, np.inf, np.inf])

        optimised_params, covariance_matrix = optimize.curve_fit(
            self._skew_norm_fit_function,
            self.x_array,
            self.y_array,
            p0=[0, estimated_mean, estimated_sigma, estimated_pdf_scaling_constant],
            maxfev=4000,
            bounds=bounds,
        )
        a, location, scale, pdf_scaling_constant = optimised_params
        mean, variance, skewness, kurtosis = skewnorm.stats(
            a, loc=location, scale=scale, moments="mvsk"
        )

        x_tmp = np.linspace(min(self.x_array), max(self.x_array), 4096)
        y_tmp = pdf_scaling_constant * skewnorm.pdf(x_tmp, a, loc=location, scale=scale)
        FWHM_left, FWHM_right, FWHM = self._full_width_at_half_maximum(x_tmp, y_tmp)

        if self.flipped_gaussian:
            peak = (x_tmp[np.argmax(y_tmp)], -1 * max(y_tmp))
        else:
            peak = (x_tmp[np.argmax(y_tmp)], max(y_tmp))

        return ScanStats1D(
            skewness=skewness,
            mean=mean,
            peak=peak,
            sigma=np.sqrt(variance),
            FWHM=FWHM,
            FWHM_x_coords=(FWHM_left, FWHM_right),
            kurtosis=kurtosis,
            skewnorm_fit_parameters=SkewNormFitParameters(
                a=a,
                location=location,
                scale=scale,
                pdf_scaling_constant=pdf_scaling_constant,
                covariance_matrix=covariance_matrix,
            ),
        )

    def _full_width_at_half_maximum(
        self, x: npt.NDArray, y: npt.NDArray
    ) -> tuple[float, float, float]:
        """
        Calculates the full width at half maximum

        Parameters
        ----------
        x : npt.NDArray
            The x array
        y : npt.NDArray
            The y array

        Returns
        -------
        tuple[float, float, float]
            The left x coordinated of the FWHM, the right x coordinate of the FWHM,
            and the FWHM
        """
        args_y = np.where(y < np.max(y) / 2)[0]

        arg_y_limit = np.where(np.diff(args_y) > 1)[0]
        if len(arg_y_limit) == 0:
            warn(
                "Full width at half maximum could not be calculated. "
                "The full distribution has most likely not been sampled"
            )
            return (None, None, None)
        arg_left = args_y[arg_y_limit][0]
        arg_right = args_y[arg_y_limit + 1][0]

        return (
            x[arg_left],
            x[arg_right],
            abs(x[arg_left] - x[arg_right]),
        )

    def _skew_norm_fit_function(
        self,
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
    stats_class = Scan1DStats(x, y)
    stats = stats_class.calculate_stats()
    print("skewness:", stats.skewness)
    print("mean:", stats.mean)
    print("sigma:", stats.sigma)
    print("FWHM:", stats.FWHM)
    print(stats.skewnorm_fit_parameters.covariance_matrix)

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
