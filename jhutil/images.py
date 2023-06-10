import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def plot_image(file_path):
    image = mpimg.imread(file_path)
    plt.imshow(image)
    plt.axis('off')
    plt.show()


def plot_images(image_path_list, num_rows=3):

    # Calculate the number of columns based on the number of rows
    num_images = len(image_path_list)
    num_cols = (num_images + num_rows - 1) // num_rows

    # Create the grid layout for plotting
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 8))

    # Iterate over the image files and plot each image
    for i, image_path in enumerate(image_path_list):
        # Calculate the row and column index for the current image
        row = i // num_cols
        col = i % num_cols

        # Load the image
        image = mpimg.imread(image_path)

        # resize image
        image = image[::10, ::10]

        # Plot the image on the corresponding subplot
        axes[row, col].imshow(image)
        axes[row, col].axis('off')

    # Adjust the spacing between subplots
    plt.tight_layout()

    # Show the plot
    plt.show()


def plot_images_in_folder(folder_path, num_rows=3, indices=None, is_print=False):

    # Get a list of all files in the folder
    files = os.listdir(folder_path)

    # Filter out non-image files
    image_files = [file for file in files if file.endswith(('.png', '.JPG', '.jpg', '.jpeg', '.gif'))]
    image_path_list = [os.path.join(folder_path, image_file) for image_file in image_files]

    if indices is not None:
        image_path_list = [image_path_list[i] for i in indices]

    if is_print:
        import jhutil; jhutil.jhprint(6666, image_path_list, list_one_line=False)

    return plot_images(image_path_list, num_rows=num_rows)
