import sys
import argparse
from ESPNN.core import run_NN


class DefaultHelpParser(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write('error: %s\n\n' % message)
        self.print_help()
        sys.exit(2)


if __name__ == "__main__":

    basic_usage = "python -m SPNN [X] [Xm] [Y] [Ym]"
    custom_ener = " [-emin EMIN] [-emax EMAX] [-npoints NPOINTS] "
    custom_output = "[--plot PLOT] [--outdir OUTDIR]"
    usage = ''.join([basic_usage, custom_ener, custom_output])
    # parser = argparse.ArgumentParser(usage=usage, add_help=False)
    parser = DefaultHelpParser(usage=usage, add_help=False)

    parser.add_argument("X", help="Projectile name", type=str)
    parser.add_argument("Xm", help="Projectile mass", type=float)
    parser.add_argument("Y", help="Target name", type=str)
    parser.add_argument("Ym", help="Target mass", type=float)
    parser.add_argument(
        "-emin",
        dest="emin",
        default=0.001,
        type=float,
        help="Minimum energy value (default: 0.001)"
    )
    parser.add_argument(
        "-emax",
        dest="emax",
        default=10.,
        type=float,
        help="Maximum energy value (default: 10)"
    )
    parser.add_argument(
        "-npoints",
        dest="npoints",
        default=1000,
        type=int,
        help="Number of grid points (default: 1000)"
    )
    parser.add_argument(
        "--plot",
        dest="plot",
        default=True,
        help="Plot prediction (default: True)"
    )
    parser.add_argument(
        "--outdir",
        dest="outdir",
        default='./',
        help="Path to output folder (default: './')"
    )
    parser.add_argument(
        "-h",
        "--help",
        action="help",
        help="Show this message and exit"
    )

    args = parser.parse_args()

    run_NN(
        projectile=args.X,
        projectile_mass=args.Xm,
        target=args.Y,
        target_mass=args.Ym,
        emin=args.emin,
        emax=args.emax,
        npoints=args.npoints,
        outdir=args.outdir,
        plot=args.plot,
    )
