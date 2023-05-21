import random
import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array
import pyopencl.clrandom as cl_random
from PIL import Image

# define radius and sigma
radius = 4
sigma = 2

# OpenCL kernel code
kernel_code = ("""
__kernel void gaussian_blur(__global const uchar4* input_image,
                            __global uchar4* output_image,
                            const int width,
                            const int height) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int index = y * width + x; \n"""
    f"const int radius = {radius}; \n" # insert radius
    f"const float sigma = {sigma}; \n" # insert sigma
    """float4 sum = 0.0f;
    float totalWeight = 0.0f;

    for (int j = -radius; j <= radius; j++) {
        for (int i = -radius; i <= radius; i++) {
            int2 offset = (int2)(i, j);
            int2 coord = (int2)(x, y) + offset;

            if (coord.x >= 0 && coord.x < width && coord.y >= 0 && coord.y < height) {
                float weight = exp(-(offset.x * offset.x + offset.y * offset.y) / (2 * sigma * sigma));
                sum += convert_float4(input_image[coord.y * width + coord.x]) * weight;
                totalWeight += weight;
            }
        }
    }

    output_image[index] = convert_uchar4(sum / totalWeight);
}
""")

# Load test image and convert it to RGBA for the blur to function properly
# Without converting it, the outcome is like in "no_RGBA_conversion.jpg"
input_image = Image.open("test_image.jpg").convert('RGBA')
width, height = input_image.size

# Convert image to numpy array
input_image_data = np.array(input_image).astype(np.uint8)

# Create OpenCL context
platform = cl.get_platforms()[0]
device = platform.get_devices()[0]
context = cl.Context([device])
queue = cl.CommandQueue(context)

# Query device max work group size and print it to the console
max_work_group_size = device.get_info(cl.device_info.MAX_WORK_GROUP_SIZE)
print(f'Maximum work group size of the device: {max_work_group_size}')

# Query device max work item sizes and print it to the console
max_work_item_sizes = device.get_info(cl.device_info.MAX_WORK_ITEM_SIZES)
print(f'Maximum work item sizes of the device: {max_work_item_sizes}')

# Create OpenCL buffers
input_image_buffer = cl_array.to_device(queue, input_image_data)
output_image_buffer = cl_array.empty_like(input_image_buffer)

# Create OpenCL program
program = cl.Program(context, kernel_code).build()

# Execute kernel
program.gaussian_blur(queue, (width, height), None,
                      input_image_buffer.data, output_image_buffer.data,
                      np.int32(width), np.int32(height))

# Read back the output image data
output_image_data = output_image_buffer.get()

# Convert output image data to PIL image
output_image = Image.fromarray(output_image_data, 'RGBA')

# generate filename, include radius and sigma in filename
randnrs = [str(random.randint(0,9)) for x in range(4)]
randnrsstr = ''.join(randnrs)
filename = f'GaussianBlur_Radius{radius}_Sigma{sigma}_{randnrsstr}.png'

#print name of image to console
print('Name of the generated image: ' + filename)

# Save output image
# needs to be a png after the RGBA conversion, since .pngs add a fourth channel: Alpha (Transparency)
output_image.save(filename)