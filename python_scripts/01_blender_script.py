"""
Blender Script for Generating Transparent Background Dataset from SVG Files

Author: Ali Tahir
Contact: contact@alitahir.dev

Disclaimer:
I am not an expert in Blender scripting or Blender itself. To create this script, 
I completed a crash course on Blender Python scripting to understand the basics and 
develop a functional script. If you're new to Blender scripting, I highly recommend 
this course as a great starting point:

Course Reference:
"Python Scripting in Blender with Practical Projects" by Thomas McDonald
Available on Udemy: https://www.udemy.com/course/python-scripting-in-blender-with-practical-projects/?couponCode=KEEPLEARNING

Feel free to modify and improve this script as needed for your own projects!
"""


import mathutils
import math
import bpy
import gc
import os

from svgpathtools import svg2paths2
from bpy_extras.object_utils import world_to_camera_view


processed_dict = {}
loaded_images = []

def cleanup_scene():
    """Fully clean up unused data to free memory."""
    global loaded_images

    # Unlink and remove all objects from collections
    for collection in bpy.data.collections:
        for obj in list(collection.objects):
            collection.objects.unlink(obj)
            bpy.data.objects.remove(obj, do_unlink=True)

    # Remove all collections
    for collection in list(bpy.data.collections):
        bpy.data.collections.remove(collection)

    # Remove all data-blocks
    data_blocks = [
        bpy.data.meshes,
        bpy.data.curves,
        bpy.data.lights,
        bpy.data.cameras,
        bpy.data.materials,
        bpy.data.textures,
        bpy.data.images,
        bpy.data.node_groups,
        bpy.data.actions,
        bpy.data.fonts,
    ]
    for data_block in data_blocks:
        for block in list(data_block):
            data_block.remove(block, do_unlink=True)

    # Remove images loaded during rendering
    for image_name in loaded_images:
        image = bpy.data.images.get(image_name)
        if image:
            bpy.data.images.remove(image, do_unlink=True)
    loaded_images.clear()

    # Purge orphaned data blocks
    for _ in range(5):
        bpy.ops.outliner.orphans_purge(do_recursive=True)

    # Force garbage collection
    gc.collect()

    print("Scene cleaned up and memory freed.")


def disable_shadow():
    for obj in bpy.data.objects:
            # Check for CURVE or MESH types
            if obj.type in {'CURVE', 'MESH'}:
                # Disable shadows for Cycles
                if hasattr(obj.cycles, 'cast_shadow'):
                    obj.cycles.cast_shadow = False

                # Disable shadows for Eevee through materials
                for slot in obj.material_slots:
                    if slot.material and hasattr(slot.material, 'shadow_method'):
                        slot.material.shadow_method = 'NONE'

def load_svg_files(files, scale=5):
    """
    Load and process SVG files into Blender.

    Args:
        files (list): List of SVG file paths to load.
        scale (int): Scaling factor for the objects.
    """
    signs_collection = bpy.data.collections.get("Signs")
    if not signs_collection:
        signs_collection = bpy.data.collections.new("Signs")
        bpy.context.scene.collection.children.link(signs_collection)

    for file in files:
        head, tail = os.path.split(file)
        bpy.ops.import_curve.svg(filepath=os.path.join(head, tail))
        ExtrudeAndScale(tail, scale)
        curve_collection = bpy.data.collections[tail]
        curve_collection.hide_render = True
        signs_collection.children.link(curve_collection)
        disable_shadow()
        bpy.data.scenes["Scene"].collection.children.unlink(curve_collection)


def batch_process_signs(files, settings_list, batch_size=5, scale=5):
    """
    Process SVG signs in smaller batches to avoid memory overload.

    Args:
        files (list): List of SVG file paths to process.
        batch_size (int): Number of signs to process in each batch.
        scale (int): Scaling factor for the objects.
    """
    total_batches = len(files) // batch_size + (1 if len(files) % batch_size != 0 else 0)
    for batch_index in range(total_batches):
        start = batch_index * batch_size
        end = min(start + batch_size, len(files))
        batch = files[start:end]

        print(f"Processing batch {batch_index + 1}/{total_batches}: Signs {start + 1} to {end}")

        # Reset the scene and recreate camera and light for each batch
        reset_existing_scene()

        # Process the current batch
        load_svg_files(batch, scale)
        set_camera_and_light(batch[0])
        CreateCircles(ver=3, hor=1) 
        AnimateCameraMovement(60)

        # Comment out the following line to disable rendering and check the scene setup and animations in blender for debugging or customizations.
        # Render the current batch
        RenderAll(output_dir, settings_list)

        # Clean up after batch
        cleanup_scene()
        print(f"Batch {batch_index + 1}/{total_batches} completed and cleaned up.")


def ExtrudeAndScale(collection, scale):
    collection_ = bpy.data.collections.get(collection)
    if not collection_:
        raise KeyError(f"Collection '{collection_}' not found. Make sure it exists.")

    for i in range(len(bpy.data.collections[collection].objects)):
        bpy.data.collections[collection].objects[i].data.extrude = 0.002 + i * 0.00001

    bpy.data.collections[collection].objects[0].select_set(True)
    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='MEDIAN')
    bpy.context.scene.cursor.location = bpy.data.collections[collection].objects[0].matrix_world.to_translation()

    for curve in bpy.data.collections[collection].objects:
        curve.select_set(True)
    bpy.ops.object.origin_set(type='ORIGIN_CURSOR', center='MEDIAN')

    bpy.context.scene.cursor.location = (0,0,0)

    for curve in bpy.data.collections[collection].objects:
        if curve != bpy.data.collections[collection].objects[0]:
            curve.parent = bpy.data.collections[collection].objects[0]
            curve.location = (0,0,0)

    curr_x, curr_y, curr_z = bpy.data.collections[collection].objects[0].dimensions
    new_x = scale * curr_x
    new_y = scale * curr_y
    bpy.data.collections[collection].objects[0].dimensions = [new_x, new_y, curr_z]
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

    bpy.data.collections[collection].objects[0].location = [0, 3, 2]
    bpy.data.collections[collection].objects[0].rotation_euler = [math.pi/2, 0, 0]
        


def CreateCircles(ver, hor):
    counter = 0
    CameraArm = "CameraArm"
    
    # Create the camera arm object
    camera_arm_obj = bpy.data.objects.new(name=CameraArm, object_data=None)
    bpy.data.collections['CameraAndLight'].objects.link(camera_arm_obj)

    # Create a new collection for circles
    collection_name = 'Circles'
    if collection_name not in bpy.data.collections:
        new_collection = bpy.data.collections.new(collection_name)
        bpy.context.scene.collection.children.link(new_collection)
    else:
        new_collection = bpy.data.collections[collection_name]

    # Setup camera object
    cam_obj = bpy.data.objects["Camera"]
    cam_obj.constraints.new('COPY_LOCATION')
    cam_obj.constraints['Copy Location'].target = camera_arm_obj

    # Create a target object for the camera to track
    target_obj = bpy.data.objects.new(name="Target", object_data=None)
    bpy.data.collections['CameraAndLight'].objects.link(target_obj)
    target_obj.location = [0, 3, 2.00]

    cam_obj.constraints.new('TRACK_TO')
    cam_obj.constraints["Track To"].target = target_obj
    cam_obj.constraints["Track To"].track_axis = 'TRACK_NEGATIVE_Z'
    cam_obj.constraints["Track To"].use_target_z = False

    # Create circles and set follow path constraints
    for i in range(ver):
        for j in range(hor + 1, 1, -1):
            adjust_camera_to_fit_svg_dynamically('CameraAndLight')
            bpy.ops.curve.primitive_bezier_circle_add(radius=j, enter_editmode=False, align='WORLD', location=(0, 3, 1+i))
            curve = bpy.context.active_object  # Get the created curve object
            curve.name = f'Circle{counter}'
            bpy.context.scene.collection.objects.unlink(curve)
            new_collection.objects.link(curve)

            # Add follow path constraint
            follow_path_constraint = camera_arm_obj.constraints.new('FOLLOW_PATH')
            follow_path_constraint.target = curve
            follow_path_constraint.name = f"Follow Path {counter}"
            follow_path_constraint.mute = True
            follow_path_constraint.keyframe_insert(data_path="mute", frame=1)
            counter += 1

        
def AnimateCameraMovement(framesPerCircle):
    arm = bpy.data.objects['CameraArm']
    counter = 1
    for constraint in bpy.data.objects['CameraArm'].constraints.values():
        constraint.mute = False
        constraint.keyframe_insert(data_path="mute", frame = counter) 
        constraint.offset = 90
        constraint.keyframe_insert(data_path="offset", frame = counter) 
        constraint.offset = 110
        constraint.keyframe_insert(data_path="offset", frame = counter + framesPerCircle)
        constraint.mute = True
        constraint.keyframe_insert(data_path="mute", frame = counter + framesPerCircle) 
        counter += framesPerCircle
    bpy.context.scene.frame_end = counter

        
def RandomLight(min, max):
    light = bpy.data.objects['Light']
    for i in range (bpy.context.scene.frame_end):
        light.data.energy = mathutils.noise.random() * (max - min) + min
        light.data.keyframe_insert("energy", frame = i)
        

def RenderAll(output_dir, settings_list):
    """
    Render images for each SVG sign with specified settings.

    Args:
        output_dir (str): The parent output folder path.
        settings_list (list): A list of dictionaries containing render settings.
        texture_folder (str): The folder containing texture files.
    """
    bpy.context.scene.render.use_overwrite = False
    bpy.context.scene.render.film_transparent = True  # Ensure transparent background

    for idx, sign in enumerate(bpy.data.collections["Signs"].children):
        print(f"Rendering sign: {sign.name}")
        sign.hide_render = False

        for settings in settings_list:
            resolution = settings["resolution"]

            # Set render resolution
            bpy.context.scene.render.resolution_x = resolution
            bpy.context.scene.render.resolution_y = resolution

            # Set output path
            processed_dict[idx] = sign.name.replace(".svg", "")
            output_path = os.path.join(output_dir, sign.name.replace(".svg", ""), "transparent")
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            bpy.context.scene.render.filepath = os.path.join(output_path, "")

            # Render animation
            bpy.context.scene.eevee.use_soft_shadows = False
            bpy.context.scene.cycles.use_soft_shadows = False

            bpy.ops.render.render(animation=True)

        sign.hide_render = True


def reset_existing_scene():
     for coll in bpy.data.collections:
         if coll:
             obs = [o for o in coll.objects]
             while obs:
                 bpy.data.objects.remove(obs.pop())
             bpy.data.collections.remove(coll)


def get_svg_bounding_box(svg_file):
    """Extract the bounding box dimensions from an SVG file."""
    paths, attributes, _ = svg2paths2(svg_file)
    min_x, min_y, max_x, max_y = float('inf'), float('inf'), float('-inf'), float('-inf')

    for path in paths:
        for segment in path:
            bbox = segment.bbox()  # Get bounding box for each path segment
            min_x, min_y = min(min_x, bbox[0]), min(min_y, bbox[1])
            max_x, max_y = max(max_x, bbox[2]), max(max_y, bbox[3])

    width = max_x - min_x
    height = max_y - min_y
    return min_x, min_y, max_x, max_y, width, height


def set_camera_and_light(svg_file):
    # Get SVG bounding box
    min_x, min_y, max_x, max_y, width, height = get_svg_bounding_box(svg_file)

    # Calculate SVG center
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2

    # Set camera distance and angle to frame the SVG with padding
    camera_distance = max(width, height)

    # Ensure the CameraAndLight collection exists
    collection_name = "CameraAndLight"
    if collection_name not in bpy.data.collections:
        new_collection = bpy.data.collections.new(collection_name)
        bpy.context.scene.collection.children.link(new_collection)
    else:
        new_collection = bpy.data.collections[collection_name]

    # Camera setup
    cam = bpy.data.cameras.new("Camera")
    cam_obj = bpy.data.objects.new("Camera", cam)
    cam_obj.location = (center_x, center_y, camera_distance)
    cam_obj.rotation_euler = (math.radians(90), 0, 0)
    cam.type = 'PERSP'
    cam.lens = 40  # Adjust lens if necessary
    new_collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj

    # Light setup
    light = bpy.data.lights.new("Light", 'SUN')  # Use SunLight type
    light.use_shadow = False
    light_obj = bpy.data.objects.new("Light", light)
    new_collection.objects.link(light_obj)
    light_obj.location = (center_x, center_y, camera_distance * 1.5)
    light_obj.rotation_euler = (math.radians(90), 0, 0)

    # Disable shadows for the light
    light_obj.data.use_shadow = False
    light_obj.data.specular_factor = 0
    light_obj.data.energy = 4.0  # Adjust energy for lighting without shadows

    # Ensure Eevee does not enable global soft shadows
    bpy.context.scene.eevee.use_soft_shadows = False

    # Debugging output
    print(f"SVG Dimensions: Width={width}, Height={height}")
    print(f"Center: X={center_x}, Y={center_y}")
    print(f"Camera and light positioned for SVG: {svg_file}")


def adjust_camera_to_fit_svg_dynamically(collection_name):
    """
    Adjust the camera's focal length and position to fit the SVG within the frame dynamically during animation,
    ensuring the gap does not exceed the required amount on each side.
    """
    collection = bpy.data.collections.get(collection_name)
    if not collection:
        raise ValueError(f"Collection '{collection_name}' not found.")

    camera = bpy.context.scene.camera
    total_frames = bpy.context.scene.frame_end

    depsgraph = bpy.context.evaluated_depsgraph_get()

    # Ensure the camera is perspective
    camera.data.type = 'PERSP'
    camera.data.lens = 70

    # Set the camera's sensor fit to 'AUTO' to handle aspect ratio automatically
    camera.data.sensor_fit = 'AUTO'

    # Store initial camera lens and distance
    initial_lens = camera.data.lens
    initial_location = camera.location.copy()
    initial_distance = (camera.location - mathutils.Vector((0.0, 0.0, 0.0))).length

    # Get render resolution to account for aspect ratio
    render = bpy.context.scene.render
    aspect_ratio = render.resolution_x / render.resolution_y

    for frame in range(1, total_frames + 1):
        bpy.context.scene.frame_set(frame)

        # Initialize min and max NDC coordinates
        min_ndc = mathutils.Vector((1.0, 1.0))
        max_ndc = mathutils.Vector((0.0, 0.0))

        for obj in collection.objects:
            if obj.type == 'CURVE':
                eval_obj = obj.evaluated_get(depsgraph)
                mesh = eval_obj.to_mesh()
                if mesh:
                    mesh.transform(eval_obj.matrix_world)
                    for vertex in mesh.vertices:
                        co_world = vertex.co
                        co_ndc = world_to_camera_view(bpy.context.scene, camera, co_world)
                        ndc_xy = mathutils.Vector((co_ndc.x, co_ndc.y))
                        min_ndc.x = min(min_ndc.x, ndc_xy.x)
                        min_ndc.y = min(min_ndc.y, ndc_xy.y)
                        max_ndc.x = max(max_ndc.x, ndc_xy.x)
                        max_ndc.y = max(max_ndc.y, ndc_xy.y)
                    eval_obj.to_mesh_clear()
                else:
                    print(f"Warning: Object {obj.name} could not be converted to mesh.")

        # Calculate how much of the frame the object occupies
        delta_ndc_x = max_ndc.x - min_ndc.x
        delta_ndc_y = max_ndc.y - min_ndc.y

        # Apply margin
        margin = 0.05  # 5% margin
        required_ndc_x = delta_ndc_x * (1 + margin)
        required_ndc_y = delta_ndc_y * (1 + margin)

        # Calculate scaling factors for X and Y
        scale_factor_x = required_ndc_x / 1.0
        scale_factor_y = required_ndc_y / 1.0

        # Adjust for aspect ratio
        if aspect_ratio >= 1.0:
            # Wide image, adjust Y scaling factor
            scale_factor = max(scale_factor_x, scale_factor_y / aspect_ratio)
        else:
            # Tall image, adjust X scaling factor
            scale_factor = max(scale_factor_x / aspect_ratio, scale_factor_y)

        # Limit the scale factor to prevent extreme zooms
        scale_factor = max(scale_factor, 0.7)  # Prevent zooming in too much
        scale_factor = min(scale_factor, 1.0)  # Prevent zooming out too much

        # Adjust the camera's focal length
        new_focal_length = initial_lens / scale_factor
        camera.data.lens = new_focal_length
        camera.data.keyframe_insert(data_path="lens", frame=frame)

        # Adjust the camera's position along its local Z-axis
        # Calculate the new distance
        new_distance = initial_distance * scale_factor
        direction = camera.matrix_world.to_quaternion() @ mathutils.Vector((0.0, 0.0, -1.0))
        camera.location = initial_location + direction * (new_distance - initial_distance)
        camera.keyframe_insert(data_path="location", frame=frame)
                

def read_svg_files(path):
    """
    Reads all SVG files from the given directory path.
    Args:
        path (str): The directory path to search for SVG files.
    Returns:
        list: A list of file paths for all SVG files in the directory.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"The path '{path}' does not exist.")
    if not os.path.isdir(path):
        raise NotADirectoryError(f"The path '{path}' is not a directory.")
    svg_files = [os.path.join(path, file) for file in os.listdir(path) if file.endswith('.svg') and 'blanko' not in file]
    return svg_files


if __name__ == "__main__":
    svg_files_dir = "C:/Users/ali/workspace/svgs"
    output_dir = "C:/Users/ali/workspace/transparent_dataset_directory"

    # Define settings for each folder
    settings_list = [
       {'resolution': 300},
    ]

    # Read all SVG files
    svg_files = read_svg_files(svg_files_dir)

    # Process in batches
    batch_process_signs(svg_files, settings_list, batch_size=1, scale=5)

    # print(json.dumps(processed_dict))