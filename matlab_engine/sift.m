function siftedFIDs = sift(fids)
    % SIFT - Processes raw FIDs using a thresholding method.
    %
    % This function processes the provided FID data by thresholding in the
    % frequency domain to remove noise and other unwanted components. The
    % function returns the processed FID data.
    %
    % Syntax:
    % siftedFIDs = sift(fids)
    %
    % Inputs:
    % fids : Matrix containing FID data. Each column represents a different FID,
    %        and each row represents a different time point.
    %
    % Outputs:
    % siftedFIDs : Matrix of processed FID data after thresholding.
    %
    % Example:
    % rawFIDs = some_function_to_generate_fids();
    % processedFIDs = sift(rawFIDs);

    % Number of samples
    npts = size(fids, 1);

    % Take indirect Fourier transform of FIDs
    dis_pfile = fids(:, 1:end - 1);
    sift_fft = fftshift(fft(dis_pfile, [], 2), 2);
    thresh_sift_fft = sift_fft;

    % Determine the noise threshold for each indirect frequency
    noise = sift_fft(:, end - round(size(sift_fft, 2) / 6):end);
    thresh = 2 * std(abs(noise')) + mean(abs(noise'));

    % Zero out all points below the threshold in the indirect frequency spectrum
    for k = 1:npts
        change = abs(sift_fft(k, :)) < thresh(k);
        thresh_sift_fft(k, change) = 0;
    end

    % Remove isolated spikes in the indirect frequency spectrum
    ts = find(abs(thresh_sift_fft) > 0);
    ts(ts > length(thresh_sift_fft)) = [];
    change2 = thresh_sift_fft(ts + 1) == 0;
    thresh_sift_fft(ts(change2)) = 0;

    % Inverse Fourier transform with respect to the indirect time domain
    siftedFIDs = ifft(ifftshift(thresh_sift_fft, 2), '', 2);

    % Compare the processed FIDs to the original and calculate residuals
    resid = real(fids(:, 1:end - 1) - siftedFIDs);
    ch = (sum(abs(resid)) >= (2 * std(resid(:))) * size(resid, 2));
    siftedFIDs(ch) = dis_pfile(ch);

    % Adjust the tail end of the FIDs
    siftedFIDs(end - 10:end, :) = siftedFIDs(end - 20:end - 10, :);

    % Re-add the last column
    siftedFIDs(:, end + 1) = fids(:, end);
end
