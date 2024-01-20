function x_filtered = filter_highpass(x)
    % FILTER_HIGHPASS Applies a highpass filter to the input data.
    %
    % Input:
    %   - x: Input data to be filtered.
    %
    % Output:
    %   - x_filtered: Filtered data.
    %
    % The function uses the highpassfilter function to design a highpass filter
    % and then applies this filter to the input data using the filtfilt function.

    % Design the highpass filter
    b = highpassfilter(length(x(:)));

    % Apply the filter to the input data
    x_filtered = filtfilt(b, 1, x(:));
end
