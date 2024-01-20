function fit_params = nls_curvefit_timefit(fitparams0, tdata, fids)
    % nls_curvefit_timefit - Non-linear least squares curve fitting for time domain signals.
    %
    % This function performs curve fitting on the mean of the provided FID data
    % using the NMR_TimeFit_v class. The function returns the fitted parameters.
    %
    % Syntax:
    % fit_params = nls_curvefit_timefit(fitparams0, tdata, fids)
    %
    % Inputs:
    % fitparams0 : Initial guess for the fit parameters. It should be a vector
    %              with 15 elements, structured as follows:
    %              [area1, area2, area3, freq1, freq2, freq3, fwhm1, fwhm2, fwhm3,
    %               fwhmG1, fwhmG2, fwhmG3, phase1, phase2, phase3]
    % tdata      : Time data corresponding to the FID data. It should be a column vector.
    % fids       : FID data. Each column represents a different FID, and each row
    %              represents a different time point.
    %
    % Outputs:
    % fit_params : Fitted parameters obtained after curve fitting. The structure is the same as fitparams0.
    %
    % Example:
    % fitparams0 = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1];
    % tdata = linspace(0, 1, 100)';
    % fids = some_function_to_generate_fids(tdata);
    % fit_params = nls_curvefit_timefit(fitparams0, tdata, fids);

    % Compute the mean of the FID data across all columns
    ydata = mean(fids, 2);

    % Initialize the NMR_TimeFit_v class with the provided data and initial guess parameters
    nmrFit = NMR_TimeFit_v(ydata(:), tdata(:), fitparams0(1:3), ...
        fitparams0(4:6), fitparams0(7:9), fitparams0(10:12), fitparams0(13:15), [], []);

    % Perform the curve fitting
    nmrFit = nmrFit.fitTimeDomainSignal();

    % Extract the fitted parameters
    fit_params = [nmrFit.area(:); nmrFit.freq(:); nmrFit.fwhm(:); nmrFit.fwhmG(:); nmrFit.phase(:)];
end
