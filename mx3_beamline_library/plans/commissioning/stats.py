import numpy as np
from scipy import optimize
from scipy.stats import skewnorm
from pydantic import BaseModel

class ScanStats1D(BaseModel):
    skewness: float
    mean: float
    maximum_y_value: float
    sigma: float
    FWHM: float
    FWHM_x_coords: tuple[float, float] | list[float]
    PDF_scaling_constant: float




def calculate_stats(x_array, y_array):

    estimated_mean =  sum(x_array*y_array)/sum(y_array)
    estimated_mode = x_array[np.argmax(y_array)]
    estimated_sigma = np.sqrt(sum(y_array*(x_array-estimated_mean)**2)/sum(y_array))
    estimated_pdf_scaling_constant = max(y_array) / max(y_array / (sum(y_array)))
    estimated_skewness_sign = estimated_mean - estimated_mode # if mean>mode, the distribution is positively skewed
    print("estimated mean:", estimated_mean)
    print("estimated mode:",  estimated_mode)
    print("estimated sigma", estimated_sigma)
    print("estimated C", estimated_pdf_scaling_constant)

    if estimated_skewness_sign > 0:
        bounds = ([-1e-5,0,0,0], [np.inf, np.inf, np.inf, np.inf])
    elif estimated_skewness_sign < 0:
        bounds = ([-np.inf,0,0,0], [1e-5, np.inf, np.inf, np.inf])


    optimised_params, covariance_matrix = optimize.curve_fit(
        _skew_norm_fit_function,
        x_array,
        y_array,
        p0=[0, estimated_mean, estimated_sigma, estimated_pdf_scaling_constant],
        maxfev=4000,
        bounds=bounds
        
    )
    skewness, mean, sigma, pdf_scaling_constant = optimised_params

    x_new = np.linspace(min(x_array), max(x_array), 4096)
    y_new = pdf_scaling_constant*skewnorm.pdf(x_new, skewness, mean, sigma)

    FWHM_left, FWHM_right, FWHM = full_width_at_half_maximum(x_new, y_new)



    return ScanStats1D(
        skewness=skewness, 
        mean=mean,
        maximum_y_value=max(y_array), 
        sigma=sigma,
        FWHM=FWHM, 
        FWHM_x_coords=(FWHM_left, FWHM_right), 
        PDF_scaling_constant=pdf_scaling_constant)

def full_width_at_half_maximum(x_array, y_array):
    args_y = np.where(y_array < np.max(y_array) / 2)[0]

    arg_y_limit = np.where(np.diff(args_y)>1)[0]

    arg_left = args_y[arg_y_limit][0]
    arg_right = args_y[arg_y_limit + 1][0]

    return x_array[arg_left], x_array[arg_right] , abs(x_array[arg_left] - x_array[arg_right] )


def _skew_norm_fit_function(x, a, loc, scale, C):
    return C*skewnorm.pdf(x, a, loc, scale)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    a= 3
    x = np.linspace(-20, 100, 100)
    y = skewnorm.pdf(x, a,loc=50,scale=9.8)

    stats = calculate_stats(x,y)
    plt.plot(x, stats.PDF_scaling_constant*skewnorm.pdf(x, stats.skewness, stats.mean, stats.sigma))
    plt.plot(x, y, linestyle="--")
    plt.axvline(stats.FWHM_x_coords[0])
    plt.axvline(stats.FWHM_x_coords[1])

    plt.xlabel("x")
    plt.ylabel("PDF")
    plt.savefig("stats")
