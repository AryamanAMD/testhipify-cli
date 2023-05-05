#from testhipify import *
from .testhipify import *
import argparse
def main():
    parser=argparse.ArgumentParser(description ='HIPIFY Cuda Samples.Please avoid and ignore samples with graphical operations')
    #group = parser.add_mutually_exclusive_group()
    parser.add_argument("-a", "--all", help='To run hipify-perl for all sample:python testhipify.py --all "[PATH TO SAMPLE FOLDER]"')
    parser.add_argument("-b", "--generate", help='Generate .hip files')
    parser.add_argument("-c", "--compile1", help='Compile .hip files')
    parser.add_argument("-d", "--compile2", help='Compile .hip files with static libraries')
    parser.add_argument("-e", "--execute", help='Execute .out files')
    parser.add_argument("-f", "--generate_all", help='Generate all .hip files')
    parser.add_argument("-g", "--compile1_all", help='Compile all .hip files')
    parser.add_argument("-i", "--compile2_all", help='Compile all .hip files with static libraries')
    parser.add_argument("-j", "--execute_all", help='Execute all .out files')
    parser.add_argument("-k", "--parenthesis_check", help='Remove last parts from cu.hip files which are out of bounds.')
    parser.add_argument("-l", "--parenthesis_check_all", help='Remove all last parts from cu.hip files which are out of bounds.')
    parser.add_argument("-n", "--nvidia_compile", help='Compile and execute via nvcc.',action='store_true')
    parser.add_argument("-p", "--patch", help='Apply all patches in src/patches',action='store_true')
    parser.add_argument("-t", "--tale", help='To run hipify-perl for single sample:python testhipify.py -t "[PATH TO SAMPLE]"')
    parser.add_argument("-x", "--remove", help='Remove any sample relating to graphical operations e.g.DirectX,Vulcan,OpenGL,OpenCL and so on.')
    parser.add_argument("-s", "--setup1", help='Configure dependencies automatically.',action='store_true')
    parser.add_argument("-v", "--setup2", help='Configure dependencies manually.',action='store_true')
    parser.add_argument("-u", "--new_samples", help='Download latest samples from Repository.',action='store_true')

    args=parser.parse_args()
    if args.tale:
        x=args.tale
        ##print(x)
        ftale(x)
    if args.all:
        y=args.all
        ##print(y)
        fall(y)
    if args.remove:
        z=args.remove
        rem(z)
    if args.generate:
        a=args.generate
        generate(a)
    if args.patch:
        apply_patches()	
    if args.nvidia_compile:
        nvidia_compilation()
    if args.compile1:
        b=args.compile1
        compilation_1(b)
    if args.compile2:
        c=args.compile2
        compilation_2(c)
    if args.execute:
        d=args.execute
        runsample(d)	
    if args.generate_all:
        a=args.generate_all
        generate_all(a)
    if args.compile1_all:
        b=args.compile1_all
        compilation_1_all(b)
    if args.compile2_all:
        c=args.compile2_all
        compilation_2_all(c)
    if args.execute_all:
        d=args.execute_all
        runsample_all(d)
    if args.parenthesis_check:
        e=args.parenthesis_check
        parenthesis_check(e)
    if args.parenthesis_check_all:
        f=args.parenthesis_check_all
        parenthesis_check_all(f)
    if args.setup1:
        setup1()	
    if args.setup2:
        setup2()	
    if args.new_samples:
        new_samples()	
					

if __name__ == "__main__":
    main()        