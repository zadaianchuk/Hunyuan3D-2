import time
import os
import glob
from pathlib import Path
import torch
from PIL import Image

from hy3dgen.rembg import BackgroundRemover
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

def process_object(object_name, base_path, output_base_path):
    # Initialize pipeline
    pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        'tencent/Hunyuan3D-2mv',
        subfolder='hunyuan3d-dit-v2-mv',
        variant='fp16'
    )
    
    # Process both start and end views
    for view in ['start', 'end']:
        # Get the first 4 images from train folder
        image_path = os.path.join(base_path, object_name, '*', view, 'train', '*.png')
        image_files = sorted(glob.glob(image_path))[:4]
        
        if not image_files:
            print(f"No images found for {object_name} {view}")
            continue
            
        # Create output directory
        output_dir = os.path.join(output_base_path, object_name, view)
        os.makedirs(output_dir, exist_ok=True)
        
        # Process each set of 4 images as different views
        if len(image_files) >= 4:
            print(f"Processing {object_name} {view} images as multi-view set")
            
            # Load all 4 images
            images = {
                "front": Image.open(image_files[0]).convert("RGBA"),
                "left": Image.open(image_files[1]).convert("RGBA"),
                "back": Image.open(image_files[2]).convert("RGBA"),
                "right": Image.open(image_files[3]).convert("RGBA")  # Using 4th image as additional view
            }
            
            # Generate mesh for the set of images
            start_time = time.time()
            mesh = pipeline(
                image=images,
                num_inference_steps=50,
                octree_resolution=380,
                num_chunks=20000,
                generator=torch.manual_seed(12345),
                output_type='trimesh'
            )[0]
            
            print(f"Generation time: {time.time() - start_time:.2f} seconds")
            
            # Save mesh
            output_path = os.path.join(output_dir, f'multiview.glb')
            mesh.export(output_path)
            print(f"Saved mesh to {output_path}")
        else:
            print(f"Not enough images found for {object_name} {view} (need at least 4)")

def main():
    # Base paths
    base_path = '/home/azadaianchuk/projects/Hunyuan3D-2/data/load/sapien'
    output_base_path = '/home/azadaianchuk/projects/Hunyuan3D-2/data/meshes'
    
    # Get all object directories
    object_dirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    
    # Process each object
    for object_name in object_dirs:
        print(f"\nProcessing {object_name}...")
        process_object(object_name, base_path, output_base_path)

if __name__ == "__main__":
    main()
