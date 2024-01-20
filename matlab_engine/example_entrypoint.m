% Example script to use nls_curvefit function

% Generate synthetic tdata and ydata
tdata = linspace(0, 1, 1000);
freqs = [5, 10, 15];
areas = [1, 0.8, 0.6];
fwhms = [0.1, 0.2, 0.3];
fwhmGs = [0.05, 0.1, 0.15];
phases = [0, 45, 90];

ydata = zeros(size(tdata));

for i = 1:length(freqs)
    ydata = ydata + areas(i) .* exp(1i * pi / 180 .* phases(i) + 1i * 2 * pi .* tdata .* freqs(i)) ...
        .* exp(-pi .* tdata .* fwhms(i));
end

ydata = ydata + 0.05 * (rand(size(tdata)) + 1i * rand(size(tdata))); % Add some noise

% Initial guess for the parameters
fitparams0 = [areas, freqs, fwhms, fwhmGs, phases];

% Set lower and upper bounds
lb = [];
ub = [];

% Call the nls_curvefit function
fit_params = nls_curvefit(fitparams0, tdata, ydata, lb, ub);
