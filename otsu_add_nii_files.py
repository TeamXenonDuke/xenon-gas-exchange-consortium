import nibabel as nib
import numpy as np

########


def combine_binary_nii_files(file1, file2, output_file):
    """
    Combines two binary NIfTI files (containing white/black pixels) by adding the values
    and saves the resulting file.

    Parameters:
    - file1: Path to the first NIfTI file.
    - file2: Path to the second NIfTI file.
    - output_file: Path to save the resulting NIfTI file.
    """
    # Load the NIfTI files
    nii1 = nib.load(file1)
    nii2 = nib.load(file2)
    
    # Get the image data as numpy arrays
    data1 = nii1.get_fdata()
    data2 = nii2.get_fdata()
    
    # Debugging: Print summaries of input data
    print("File 1 data shape:", data1.shape)
    print("File 1 data non-zero count:", np.count_nonzero(data1))
    print("File 1 data unique values:", np.unique(data1))
    
    print("File 2 data shape:", data2.shape)
    print("File 2 data non-zero count:", np.count_nonzero(data2))
    print("File 2 data unique values:", np.unique(data2))
    
    # Check if dimensions match
    if data1.shape != data2.shape:
        raise ValueError("The dimensions of the two NIfTI files do not match!")
    
    # Check if affine matrices match
    if not np.allclose(nii1.affine, nii2.affine):
        print("Aligning second file to the first file's affine space...")
        nii2 = nib.Nifti1Image(data2, affine=nii1.affine, header=nii1.header)
        data2 = nii2.get_fdata()
    
    # Ensure binary values
    data1 = np.clip(data1, 0, 1)
    data2 = np.clip(data2, 0, 1)
    
    # Combine the binary data
    result_data = data1 + data2
    print("Result data max value:", np.max(result_data))
    print("Result data non-zero count:", np.count_nonzero(result_data))
    print("Result data unique values:", np.unique(result_data))
    
    # Ensure output is binary
    result_data = np.clip(result_data, 0, 1)
    
    # Save the combined result
    result_nii = nib.Nifti1Image(result_data, affine=nii1.affine, header=nii1.header)
    nib.save(result_nii, output_file)
    print(f"Combined binary NIfTI saved to {output_file}")

# Example usage
SUBJECT = "TY-003"
file1 = f"data/TY_subjects/{SUBJECT}/{SUBJECT}_mask_trachea_lung_corrected.nii"
file2 = f"data/TY_subjects/{SUBJECT}/mask_reg_corrected.nii"
output_file = f"data/TY_subjects/{SUBJECT}/{SUBJECT}_mask_big_corrected.nii"

combine_binary_nii_files(file1, file2, output_file)