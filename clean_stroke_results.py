import os
import shutil

def copy_filtered_images(root_dir):
    # Get absolute path of the original root directory
    abs_root_dir = os.path.abspath(root_dir)
    base_dir, root_name = os.path.split(abs_root_dir)
    
    # Create new base directory name by appending '_clean' to the original name
    new_root_name = root_name + '_clean'
    new_root_dir = os.path.join(base_dir, new_root_name)
    
    # Create the new root directory if it doesn't already exist
    if not os.path.exists(new_root_dir):
        os.makedirs(new_root_dir)
        print(f"Created directory: {new_root_dir}")
    
    # Walk through the original directory structure
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Compute the relative path to preserve folder structure
        rel_path = os.path.relpath(dirpath, root_dir)
        dest_dir = os.path.join(new_root_dir, rel_path)
        
        # Create destination directory if it doesn't exist
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)
            print(f"Created directory: {dest_dir}")
        
        # Process each file in the current directory
        for filename in filenames:
            # Only process .png files
            if not filename.endswith('.png'):
                continue

            lower_filename = filename.lower()
            # Condition 1: Files containing "start_fmri" with either "iter_240" or "iter_600"
            condition1 = ("start_fmri" in lower_filename and 
                          "size_large" in lower_filename and
                          ("iter_400" in lower_filename or "iter_1600" in lower_filename))
            # Condition 2: Files containing "original_image" or "normal_fmri"
            condition2 = ("original_image" in lower_filename or "normal_fmri" in lower_filename)
            
            if condition1 or condition2:
                src_file = os.path.join(dirpath, filename)
                dst_file = os.path.join(dest_dir, filename)
                shutil.copy2(src_file, dst_file)
                print(f"Copied: {src_file} -> {dst_file}")

if __name__ == '__main__':
    # Set the path to your original root directory here
    root_directory = '/home/jonathak/VisualEncoder/DIP_decoder/stroke_experiment_results_predicted_fMRI'
    
    # Run the copying function
    copy_filtered_images(root_directory)
