import numpy as np
import cv2
import glob
import pickle
from skimage.measure import label, regionprops
from scipy.ndimage import binary_dilation, distance_transform_edt
import matplotlib.pyplot as plt
from skimage.transform import resize
import pandas as pd
import cairo


def generate_new_color(existing_colors, seed=None):
    """
    Generate a new random color that is not in the existing_colors array.
    """
    np.random.seed(seed)
    while True:
        # Generate a random color (R, G, B)
        new_color = np.random.randint(0, 256, size=3)
        # Ensure the color is not black and not already in existing_colors
        if not np.any(np.all(existing_colors == new_color, axis=1)) and not np.all(new_color == [0, 0, 0]):
            return new_color


def capsule_path(canvas, l, r):
    path = canvas.beginPath()
    path.moveTo(-l / 2.0, -r)
    path.lineTo(l / 2.0, -r)

    path.arcTo(l / 2.0 - r, -r, l / 2.0 + r, r, -90, 180)

    path.lineTo(-l / 2.0, r)
    path.arc(-l / 2.0 - r, -r, -l / 2.0 + r, r, 90, 180)
    # path.close()
    return path


def draw_capsule(ctx, p, d, l, r, color):
    """
    CellModeller-accurate capsule drawing using Cairo.
    p = (x,y) position FLOAT
    d = (dx,dy) direction FLOAT (unit vector)
    l = length
    r = half-width (radius)
    color = (R,G,B) 0–255
    """

    # Convert color to Cairo 0–1 range
    R, G, B = [c / 255.0 for c in color]
    ctx.save()

    # Move origin to p
    ctx.translate(p[0], p[1])

    # Rotate to align with direction vector
    angle = np.degrees(np.atan2(d[1], d[0]))
    ctx.rotate(np.radians(angle))

    # Begin path
    ctx.new_path()

    # Rectangle part around centerline
    ctx.move_to(-l / 2, -r)
    ctx.line_to(l / 2, -r)
    ctx.arc(l / 2, 0, r, -np.pi / 2, np.pi / 2)  # right cap
    ctx.line_to(-l / 2, r)
    ctx.arc(-l / 2, 0, r, np.pi / 2, -np.pi / 2)  # left cap

    # Fill
    ctx.set_source_rgb(R, G, B)
    ctx.fill()

    ctx.restore()


def draw_bacteria_on_array(sel_time_point_df, colors, ctx, x_min_val, y_min_val, margin=50):
    """
    Draw the bacteria on a numpy array (image) using given colors.
    """
    for idx, bacterium in sel_time_point_df.iterrows():
        p = np.array([bacterium['Location_Center_X'], bacterium['Location_Center_Y']])
        bac_orientation = bacterium['AreaShape_Orientation']
        bac_orientation = -(bac_orientation + 90) * np.pi / 180
        d = np.array([np.cos(bac_orientation), np.sin(bac_orientation)])
        l = bacterium['AreaShape_MajorAxisLength']
        r = bacterium['AreaShape_MinorAxisLength']

        # Adjust the endpoints based on minimum x and y values and a margin
        p[0] = p[0] - x_min_val + margin
        p[1] = p[1] - y_min_val + margin

        color = tuple(int(c) for c in colors[idx])  # Ensure color is a tuple of integers
        draw_capsule(ctx, p, d, l, r, color)


def downscale_image(image_array, scale_factor):
    """
    Downscale the image by a factor to reduce its size for faster processing.
    """
    return resize(image_array,
                  (image_array.shape[0] // scale_factor, image_array.shape[1] // scale_factor),
                  order=0, anti_aliasing=False, preserve_range=True).astype(np.uint8)


def upscale_labels(labeled_array, original_shape, scale_factor):
    """
    Upscale the labeled array back to the original image shape after processing.
    """
    return resize(labeled_array,
                  original_shape,
                  order=0, anti_aliasing=False, preserve_range=True).astype(np.int32)


def fast_color_to_label(image_array, colors, scale_factor=10):
    """
    Convert the image array's colors to unique labels using downscaling for faster processing.
    Each unique color is mapped to a unique label.
    """
    # Downscale the image to reduce processing time
    downscaled_image = downscale_image(image_array, scale_factor)

    # Initialize the labeled array for the downscaled image
    downscaled_labeled_array = np.zeros(downscaled_image.shape[:2], dtype=np.int32)

    # Label each color in the downscaled image
    for label_id, color in enumerate(colors, start=1):
        # Create a mask where the color matches
        mask = np.all(downscaled_image == color, axis=-1)
        # Apply the label to the masked regions
        downscaled_labeled_array[mask] = label_id

    # Upscale the labeled array back to the original image size
    labeled_array = upscale_labels(downscaled_labeled_array, image_array.shape[:2], scale_factor)

    return labeled_array


def fast_expand_labels(labeled_array):
    """
    Expand labels until they touch each other using the distance transform.
    """
    # Compute the distance transform and nearest label indices
    distances, nearest_label = distance_transform_edt(labeled_array == 0, return_indices=True)

    # Create a copy of the labeled array for expansion
    expanded_labels = labeled_array.copy()

    # Assign the nearest labels to expand the regions
    expanded_labels[distances > 0] = labeled_array[tuple(nearest_label[:, distances > 0])]

    return expanded_labels


def create_cairo_surface(width, height):
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
    ctx = cairo.Context(surface)

    # Disable antialiasing → crucial for exact RGB matching
    ctx.set_antialias(cairo.ANTIALIAS_NONE)

    # White background
    ctx.set_source_rgb(0, 0, 0)
    ctx.paint()

    # Create NumPy view
    buf = surface.get_data()
    img = np.ndarray(shape=(height, width, 4),
                     dtype=np.uint8,
                     buffer=buf)

    return surface, ctx, img


def cairo_to_rgb(image):
    # image is BGRA (Cairo order)
    b = image[:, :, 0]
    g = image[:, :, 1]
    r = image[:, :, 2]
    rgb = np.stack([r, g, b], axis=2)
    return rgb


def find_neighbors(sel_time_point_df, time, rows, margin=50):
    # List to hold the rows of the dataframe

    # Generate unique colors for each bacterium
    existing_colors = np.array([[0, 0, 0]])
    colors = []
    for _ in range(sel_time_point_df.shape[0]):
        new_color = generate_new_color(existing_colors)
        colors.append(new_color)
        existing_colors = np.vstack([existing_colors, new_color])

    # Determine the minimum and maximum x and y values
    x_min_val = min(sel_time_point_df['Node_x1_x'].min(), sel_time_point_df['Node_x2_x'].min())
    y_min_val = min(sel_time_point_df['Node_x1_y'].min(), sel_time_point_df['Node_x2_y'].min())

    max_x = max(sel_time_point_df['Node_x1_x'].max(), sel_time_point_df['Node_x2_x'].max())
    max_y = max(sel_time_point_df['Node_x1_y'].max(), sel_time_point_df['Node_x2_y'].max())

    # Calculate the dimensions of the image
    image_width = int(max_x - x_min_val + 2 * margin)
    image_height = int(max_y - y_min_val + 2 * margin)
    surface, ctx, image = create_cairo_surface(image_width, image_height)

    # Draw bacteria on the image array using their assigned colors
    draw_bacteria_on_array(sel_time_point_df, colors, ctx, x_min_val, y_min_val, margin=margin)
    image_array = cairo_to_rgb(image)
    image_array_corrected = np.flipud(image_array)

    # plt.figure(figsize=(8, 8))
    # plt.imshow(image_array_corrected)
    # plt.axis('off')
    # plt.title("Rendered Bacteria Capsules")
    # plt.savefig(f"img/{time}.jpg", dpi=600)
    # plt.close()
    # breakpoint()

    # Efficiently label the image based on unique colors using downscaling
    labeled_array = fast_color_to_label(image_array, colors)
    # print("Any labels?", np.any(labeled_array))
    # unique_colors = np.unique(image_array.reshape(-1, 3), axis=0)
    # print(unique_colors)
    # breakpoint()

    # Expand the labels until they touch
    expanded_labels = fast_expand_labels(labeled_array)

    # Initialize the expanded image array for visualization
    expanded_image_array = np.zeros_like(image_array)

    # Convert colors to numpy array for easy indexing
    colors_array = np.array(colors)

    # Apply the colors to the expanded regions
    mask = expanded_labels > 0
    expanded_image_array[mask] = colors_array[expanded_labels[mask] - 1]

    # Display the expanded objects together (commented out for script use)
    # plt.figure(figsize=(8, 8))
    # plt.imshow(expanded_image_array)
    # plt.axis('off')
    # plt.title("Expanded Bacteria Coloring")
    # plt.show()
    # breakpoint()

    # Identify and report neighboring objects by ID
    id_map = {idx + 1: bacterium['id'] for idx, bacterium in sel_time_point_df.iterrows()}
    neighbors = {}
    for region in regionprops(expanded_labels):
        label_id = region.label

        # Create full-sized mask for this label
        region_mask = expanded_labels == label_id

        # Dilate it to touch neighbors
        expanded_mask = binary_dilation(region_mask)

        # Find labels in the dilated region
        neighboring_labels = np.unique(expanded_labels[expanded_mask])

        # Remove background and self
        neighboring_labels = neighboring_labels[(neighboring_labels != 0) & (neighboring_labels != label_id)]

        neighbors[id_map[label_id]] = [id_map[n] for n in neighboring_labels]

    # Print the neighboring objects by ID
    for bacterium_id, neighbor_list in neighbors.items():
        for neighbor_id in neighbor_list:
            rows.append((time, bacterium_id, neighbor_id))

    return rows


def neighbor_finders(bacteria_properties_df):
    rows = []

    for time_point in bacteria_properties_df['ImageNumber'].unique():
        sel_time_point_df = bacteria_properties_df.loc[
            bacteria_properties_df['ImageNumber'] == time_point].reset_index(drop=True)

        rows = find_neighbors(sel_time_point_df, time_point, rows)

    # Create a dataframe from the list of tuples
    df = pd.DataFrame(rows, columns=['Image Number', 'First Object id', 'Second Object id'])
    return df
