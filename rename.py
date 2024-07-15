import os

def rename_images_in_folder(folder_path):
    # Get a list of files in the folder
    files = os.listdir(folder_path)
    
    # Filter out files that are not images (optional: adjust as needed for specific image types)
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff'}
    image_files = [file for file in files if os.path.splitext(file)[1].lower() in image_extensions]
    
    # Sort the image files (optional: adjust sorting if needed)
    image_files.sort()

    # Rename the images
    for index, filename in enumerate(image_files, start=1):
        # Get the file extension
        file_extension = os.path.splitext(filename)[1]
        
        # Create the new filename
        new_filename = f"{index}{file_extension}"
        
        # Construct the full file paths
        src = os.path.join(folder_path, filename)
        dst = os.path.join(folder_path, new_filename)
        
        # Rename the file
        os.rename(src, dst)
        print(f"Renamed '{filename}' to '{new_filename}'")

# Usage example:
folder_path = r'C:\Users\johns\Downloads\51000-20240626T222743Z-002\51000'
rename_images_in_folder(folder_path)
