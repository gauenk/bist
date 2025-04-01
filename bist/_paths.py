# Configure paths for your computer to enable datasets

BIST_HOME="/home/gauenk/Documents/packages/bist/"
SEGTRACKv2_ROOT = "/home/gauenk/Documents/packages/superpixel-benchmark/docker/in/SegTrackv2/"
DAVIS_ROOT = "/home/gauenk/Documents/data/davis/DAVIS/"
if SEGTRACKv2_ROOT == "" or DAVIS_ROOT == "":
    print("Please set SEGTRACKv2_ROOT and DAVIS_ROOT in bist/_paths.py")
