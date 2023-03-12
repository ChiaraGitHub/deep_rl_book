import math
from tensorboardX import SummaryWriter

# NB: the folder runs is created!
# To monitor in tensorboard: tensorboard --logdir runs

if __name__ == "__main__":

    # This is to monitor values in tensorboard
    writer = SummaryWriter()

    # Dictionary of functions
    funcs = {"sin": math.sin, "cos": math.cos, "tan": math.tan}

    # For every angle
    for angle in range(-360, 360):

        # Convert angle to radiants
        angle_rad = angle * math.pi / 180

        # For every function apply it to the angle and add scalar to the writer
        for name, fun in funcs.items():
            val = fun(angle_rad)
            
            # Add values to the writer for monitoring
            writer.add_scalar(name, val, angle)

    writer.close()
