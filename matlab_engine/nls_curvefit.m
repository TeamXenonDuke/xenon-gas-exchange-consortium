function fit_params = nls_curvefit(fitparams0, tdata, ydata)
    % NLS_CURVEFIT - Non-linear least squares curve fitting for time-domain Voigt signals.
    %
    % This function performs a non-linear least squares curve fitting on
    % time-domain signals.
    %
    % Syntax:
    % fit_params = nls_curvefit(fitparams0, tdata, ydata)
    %
    % Inputs:
    % fitparams0 : Initial guess. It should be a vector
    %              with 15 elements, structured as follows:
    %              [area1, area2, area3, freq1, freq2, freq3, fwhm1, fwhm2, fwhm3,
    %               fwhmG1, fwhmG2, fwhmG3, phase1, phase2, phase3]
    % tdata      : Time data points. It should be a column vector
    % ydata      : Complex-valued time-domain signal data. It should be a column vector.
    %
    % Outputs:
    % fit_params : Optimized fit parameters after the curve fitting process.
    %
    % Example:
    % t = linspace(0, 1, 100);
    % y = complex(sin(2*pi*t), cos(2*pi*t));
    % fitparams0 = rand(15, 1);
    % params = nls_curvefit(fitparams0, t, yb);

    % Set optimization options for 'lsqcurvefit'
    fitoptions = optimoptions('lsqcurvefit');
    fitoptions.Display = 'off';
    fitoptions.MaxIter = 10000;
    fitoptions.TolFun = 1E-900;
    fitoptions.TolX = 1E-10;
    fitoptions.FinDiffType = 'central';
    fitoptions.Algorithm = 'trust-region-reflective';
    fitoptions.MaxFunEvals = 30000;

    % Reshape tdata and ydata for fitting both real and imaginary parts
    tdata_mod = repmat(tdata(:), [2, 1]);
    ydata_mod = [real(ydata(:)); imag(ydata(:))];

    % Perform the curve fitting
    [fit_params, ~, ~, ~, ~, ~, ~] = lsqcurvefit(@constrainedTimeSignal, ...
        fitparams0, tdata_mod, ydata_mod, [], [], fitoptions);

    function y_fit = constrainedTimeSignal(fitparams, tdata_mod)
        % CONSTRAINEDTIMESIGNAL - Internal function to compute the time-domain signal.
        %
        % This function computes the time-domain signal based on Voigt model.

        % Extract individual parameters from fitparams
        area = fitparams(1:3);
        freq = fitparams(4:6);
        fwhm = fitparams(7:9);
        fwhmG = fitparams(10:12);
        phase = fitparams(13:15);

        t = tdata_mod(1:end / 2);

        % Calculate the time-domain signal
        nTimePoints = length(t);
        nComp = length(area);
        timeDomainSignal = zeros(nTimePoints, 1) + 1j * zeros(nTimePoints, 1);

        for iComp = 1
            timeDomainSignal = timeDomainSignal + ...
                area(iComp) .* exp(1i * pi / 180 .* phase(iComp) + 1i * 2 * pi .* t .* freq(iComp)) ...
                .* exp(-pi .* t .* fwhm(iComp));
        end

        for iComp = 2:nComp
            timeDomainSignal = timeDomainSignal + ...
                area(iComp) .* exp(1i * pi / 180 .* phase(iComp) + 1i * 2 * pi .* t .* freq(iComp)) ...
                .* exp(-t .^ 2 * 4 * log(2) .* (fwhmG(iComp)) .^ 2) .* exp(-pi .* t .* fwhm(iComp));
        end

        % Separate real and imaginary parts
        y_fit = [real(timeDomainSignal(:)); imag(timeDomainSignal(:))];
    end

end
