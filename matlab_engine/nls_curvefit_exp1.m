function fit_params = nls_curvefit_sine(ydata, xdata)
    [area_fit, ~] = fit(xdata(:), ydata(:), 'exp1');
    fit_params = coeffvalues(area_fit);
end
