import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import numpy as np


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


def draw_match_from_path(img_path1, img_path2, corr, random_color=False, bold=False):
    corr1 = corr[:, 0:2]
    corr2 = corr[:, 2:]
    assert len(corr1) == len(corr2)
    assert corr1.shape[1] == 2 and corr1.shape[1] == 2

    img1, img2 = cv2.imread(img_path1), cv2.imread(img_path2)
    draw_match(img1, img2, corr1, corr2, random_color=random_color, bold=bold)
    


def draw_match(img1, img2, corr1, corr2, inlier=[True], random_color=False, radius1=1, radius2=1, resize=None, bold=False):
    """draw correspondence"""
    if resize is not None:
        scale1, scale2 = [img1.shape[1] / resize[0], img1.shape[0] /
                          resize[1]], [img2.shape[1] / resize[0], img2.shape[0] / resize[1]]
        img1, img2 = cv2.resize(img1, resize, interpolation=cv2.INTER_AREA), cv2.resize(
            img2, resize, interpolation=cv2.INTER_AREA)
        corr1, corr2 = corr1 / np.asarray(scale1)[np.newaxis], corr2 / np.asarray(scale2)[np.newaxis]
    corr1_key = [cv2.KeyPoint(corr1[i, 0], corr1[i, 1], radius1) for i in range(corr1.shape[0])]
    corr2_key = [cv2.KeyPoint(corr2[i, 0], corr2[i, 1], radius2) for i in range(corr2.shape[0])]

    assert len(corr1) == len(corr2)

    draw_matches = [cv2.DMatch(i, i, 0) for i in range(len(corr1))]
    if random_color:
        color = [(np.random.randint(100, 255), np.random.randint(100, 255), np.random.randint(100, 255))
                 for _ in range(len(corr1))]
    else:
        color = [(0, 255, 0) if cur_inlier else (0, 0, 255) for cur_inlier in inlier]
    if len(color) == 1 and not bold:
        display = cv2.drawMatches(img1, corr1_key, img2, corr2_key, draw_matches, None,
                                  matchColor=color[0],
                                  single_aspanPointColor=color[0],
                                  flags=4
                                  )
    elif len(color) == 1 and bold:
        height, width = max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1]
        display = np.zeros([height, width, 3], np.uint8)
        display[:img1.shape[0], :img1.shape[1]] = img1
        display[:img2.shape[0], img1.shape[1]:] = img2
        color = color * len(corr1)
        for i in range(len(corr1)):
            left_x, left_y, right_x, right_y = int(corr1[i][0]), int(
                corr1[i][1]), int(corr2[i][0] + img1.shape[1]), int(corr2[i][1])
            cur_color = (int(color[i][0]), int(color[i][1]), int(color[i][2]))
            cv2.line(display, (left_x, left_y), (right_x, right_y), cur_color, 10, lineType=cv2.LINE_AA)
    else:
        line_width = 10 if bold else 1
        height, width = max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1]
        display = np.zeros([height, width, 3], np.uint8)
        display[:img1.shape[0], :img1.shape[1]] = img1
        display[:img2.shape[0], img1.shape[1]:] = img2
        for i in range(len(corr1)):
            left_x, left_y, right_x, right_y = int(corr1[i][0]), int(
                corr1[i][1]), int(corr2[i][0] + img1.shape[1]), int(corr2[i][1])
            cur_color = (int(color[i][0]), int(color[i][1]), int(color[i][2]))
            cv2.line(display, (left_x, left_y), (right_x, right_y), cur_color, line_width, lineType=cv2.LINE_AA)
    
    plt.imshow(display[:,:,::-1])  # RGB-> BGR
    plt.show()
