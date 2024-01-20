function b = highpassfilter(dataLength)
    % HIGHPASSFILTER Designs a highpass filter using the FIR1 function.
    %
    % Input:
    %   - dataLength: Length of the data to be filtered.
    %
    % Output:
    %   - b: Filter coefficients.
    %
    % The function returns the filter coefficients for a highpass filter
    % designed using a Gaussian window. The filter is designed based on the
    % provided data length.

    % Define constants
    Fs = 50; % Sampling Frequency (Hz)
    Fc = 0.5; % Cutoff Frequency (Hz)
    Alpha = 2.5; % Gaussian Window Parameter
    flag = 'scale'; % Sampling Flag

    % Calculate the filter order
    N = 2 * floor(dataLength / 2/3) - 2;

    % Create the Gaussian window vector
    win = gausswin(N + 1, Alpha);

    % Calculate the filter coefficients using the FIR1 function
    b = fir1(N, Fc / (Fs / 2), 'high', win, flag);
end
