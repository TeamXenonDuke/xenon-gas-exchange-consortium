function fit_params = nls_curvefit_sine(ydata, xdata)
    % NLS_CURVEFIT_SINE Performs a nonlinear least squares curve fitting using a sine function.
    %
    % Input:
    %   - ydata: Dependent data to be fitted.
    %   - xdata: Independent data.
    %
    % Output:
    %   - fit_params: Parameters of the fitted sine curve.
    %
    % The function uses MATLAB's fit function with a sine type (sin1) and
    % the NonlinearLeastSquares method to fit the data.

    % Define the sine fit type
    ft = fittype('sin1');

    % Set up fitting options
    opts = fitoptions('Method', 'NonlinearLeastSquares');
    opts.Lower = [0 0 -Inf];
    opts.Upper = [Inf Inf Inf];

    % Perform the fit
    area_fit = fit(xdata(:), ydata(:), ft, opts);

    % Extract the fit parameters
    fit_params = coeffvalues(area_fit);
end
