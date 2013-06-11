import argparse

parser = argparse.ArgumentParser(description="Run the HUSC functions.")
subpar = parser.add_subparsers()

stitch = subpar.add_parser('stitch', 
                            help="Stitch four quadrants into one image.")
stitch.add_argument('quadrant_image', nargs=4,
                     help="The images for each quadrant in order: NW, NE, " +
                           "SW, SE.")
stitch.add_argument('output_image',
                     help="The filename for the stitched image.")

illum = subpar.add_parser('illum',
                          help="Estimate and correct illumination.")
illum.add_argument('images', nargs='+',
                   help="The input images.")
illum.add_argument('-o', '--output-suffix',
                   default='.illum.tif', metavar='SUFFIX',
                   help="What suffix to attach to the corrected images.")


def main():
    """Fetch commands from the command line."""
    args = parser.parse_args()
    print args


if __name__ == '__main__':
    main()

