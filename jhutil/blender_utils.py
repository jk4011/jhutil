
def get_fov():
    import math
    import bpy

    # Get the active camera
    camera = bpy.context.scene.camera.data

    # Get the sensor width and height, and focal length
    sensor_width = camera.sensor_width
    sensor_height = camera.sensor_height
    focal_length = camera.lens

    # Calculate horizontal FOV (FOV_X)
    fov_x_rad = 2 * math.atan(sensor_width / (2 * focal_length))
    # fov_x_deg = math.degrees(fov_x_rad)

    # Calculate vertical FOV (FOV_Y)
    fov_y_rad = 2 * math.atan(sensor_height / (2 * focal_length))
    # fov_y_deg = math.degrees(fov_y_rad)

    return fov_x_rad, fov_y_rad

