function [fit_params, snr_dyn] = nls_curvefit(fitparams0, tdata, ydata)
    % nls_curvefit - Non-linear least squares curve fitting for multiple time domain signals.
    %
    % This function performs curve fitting on each column of the provided FID data
    % using the NMR_TimeFit_v class. The function returns the fitted parameters for each column.
    %
    % Syntax:
    % fit_params = nls_curvefit(fitparams0, tdata, ydata)
    %
    % Inputs:
    % fitparams0 : Initial guess for the fit parameters. It should be a vector
    %              with 15 elements, structured as follows:
    %              [area1, area2, area3, freq1, freq2, freq3, fwhm1, fwhm2, fwhm3,
    %               fwhmG1, fwhmG2, fwhmG3, phase1, phase2, phase3]
    % tdata      : Time data corresponding to the FID data. It should be a column vector.
    % ydata      : FID data. Each column represents a different FID, and each row
    %              represents a different time point.
    %
    %
    % Outputs:
    % fit_params : Matrix of fitted parameters obtained after curve fitting. Each row corresponds
    %              to the fitted parameters for a column in ydata. The structure of parameters in each row is the same as fitparams0.
    % snr_dyn   : SNR of the fitted signal for each column of ydata.
    %
    % Example:
    % fitparams0 = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1];
    % tdata = linspace(0, 1, 100)';
    % ydata = some_function_to_generate_multiple_fids(tdata);
    % fit_params = nls_curvefit(fitparams0, tdata, ydata);

    % Initialize the output matrix
    fit_params = zeros(size(ydata, 2), 15);
    snr_dyn = zeros(size(ydata, 2), 1);
    fitparams0 = fitparams0(:);

    % Parallel loop to fit each column of ydata
    parfor iTimePoint = 1:size(ydata, 2)
        nmrFit = NMR_TimeFit_v(ydata(:, iTimePoint), tdata(:), fitparams0(1:3), ...
            fitparams0(4:6), fitparams0(7:9), fitparams0(10:12), fitparams0(13:15), [], []);
        nmrFit = nmrFit.fitTimeDomainSignal();
        fit_params_cur = [nmrFit.area(:); nmrFit.freq(:); nmrFit.fwhm(:); nmrFit.fwhmG(:); nmrFit.phase(:)];
        fit_params(iTimePoint, :) = fit_params_cur;

        % Calculate SNR
        fittedSignal = nmrFit.calcTimeDomainSignal(tdata(:));
        snr_dyn(iTimePoint) = snr(fittedSignal, ydata(:, iTimePoint) - fittedSignal);
    end

end
