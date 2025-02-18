import os
import glob
import shutil

def filter_dataset(
    original_dir: str,
    output_dir: str,
    valid_label_range=range(0, 40),
    label_ext=".txt",
    image_ext=".jpg"
):
    """
    labels_dir        : Directory containing label files.
    images_dir        : Directory containing image files.
    valid_label_range : The valid range of class IDs (default: 0..39).
    label_ext         : Extension for label files (default: ".txt").
    image_ext         : Extension for image files (default: ".jpg").
    """

    # Create an output directory for filtered labels/images if you don’t want 
    # to overwrite your original dataset. If you want to do in-place filtering,
    # skip creating these directories and just modify in place.
    labels_dir = os.path.join(original_dir, 'labels')
    images_dir = os.path.join(original_dir, 'images')
    output_labels_dir = os.path.join(output_dir, 'labels')
    output_images_dir = os.path.join(output_dir, 'images')
    
    os.makedirs(output_labels_dir, exist_ok=True)
    os.makedirs(output_images_dir, exist_ok=True)

    # Get a list of all label files
    label_files = glob.glob(os.path.join(labels_dir, f"*/*{label_ext}"))
    print(f"Original data num: {len(label_files)}")
    valid_cnt = 0
    for label_file in label_files:
        split_name = label_file.split('/')[-2]
        
        os.makedirs(os.path.join(output_labels_dir, split_name), exist_ok=True)
        os.makedirs(os.path.join(output_images_dir, split_name), exist_ok=True)
        
        base_name = label_file.split('/')[-1][:-4]
        
        image_file = os.path.join(images_dir, split_name, base_name + image_ext)

        # Read lines and filter
        with open(label_file, "r") as lf:
            lines = lf.readlines()

        if len(lines) == 1 and not lines[0].strip(): # originally background only images
            breakpoint()
            # Write filtered label file to the "filtered" directory
            filtered_label_path = os.path.join(output_labels_dir, split_name, base_name + label_ext)
            with open(filtered_label_path, "w") as lf:
                lf.write("")

            # Copy image to the filtered images directory
            if os.path.exists(image_file):
                filtered_image_path = os.path.join(output_images_dir, split_name, base_name + image_ext)
                shutil.copy2(image_file, filtered_image_path)
            else:
                print(f"Warning: No corresponding image found for {label_file}")
            valid_cnt += 1
            continue
        
        valid_lines = []
        for line in lines:
            line_stripped = line.strip()
            if not line_stripped:
                continue  # Skip empty lines if any

            # Split line and parse the first token as class ID
            parts = line_stripped.split()
            class_id = int(parts[0])
            
            if class_id in valid_label_range:
                valid_lines.append(line_stripped)

        # If we have valid lines left, write them to a new label file 
        # and copy the corresponding image
        if valid_lines:
            # Write filtered label file to the "filtered" directory
            filtered_label_path = os.path.join(output_labels_dir, split_name, base_name + label_ext)
            with open(filtered_label_path, "w") as lf:
                lf.write("\n".join(valid_lines) + "\n")

            # Copy image to the filtered images directory
            if os.path.exists(image_file):
                filtered_image_path = os.path.join(output_images_dir, split_name, base_name + image_ext)
                shutil.copy2(image_file, filtered_image_path)
            else:
                print(f"Warning: No corresponding image found for {label_file}")
            valid_cnt += 1
        else:
            # No valid labels remain. We do NOT copy the image or label file.
            # If you want to remove the original files in-place, do so here:
            # os.remove(label_file)
            # if os.path.exists(image_file):
            #     os.remove(image_file)
            pass

    print("Filtering complete!")
    print(f"Filtered datanum: {valid_cnt}")


if __name__ == "__main__":
    # Example usage:
    original_path = "./datasets/coco"
    output_path = "./datasets/coco_40"
    
    # Call the function
    filter_dataset(
        original_path,
        output_path,
        valid_label_range=range(0, 40),  # keep classes 0..39
        label_ext=".txt",
        image_ext=".jpg"
    )
