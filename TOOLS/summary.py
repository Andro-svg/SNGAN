#!/home/super/anaconda2/bin/python
from myfid import *
import my_inception_score as IS
import os
import argparse
import sys
import commands

if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument('gen_script', type=str)
    parser.add_argument('path', type=str)

    parser.add_argument('--outdir', type=str, default='./__out')
    parser.add_argument('--logfile', type=str, default='./summary_vals.txt')
    args = parser.parse_args()

    Mdls = os.listdir(args.path)
    logF = open(os.path.join(args.path, args.logfile),"a+")
    outdir = os.path.join(args.path, args.outdir)

    for filename in Mdls:
        print(filename)
        pathname = os.path.join(args.path, filename)
        script = os.path.join(args.path, args.gen_script)

        #if not os.path.isfile(pathname):
            #continue
        print("Generating Pics..")
        (status, output) = commands.getstatusoutput('python %s %s'%(script,pathname))
        if status !=0:
            print( output)
            raise (OSError,"Gen script Error")
        print("calculating FID value..")
        fid_value = calculate_fid_given_paths([cifar5000_path,args.outdir],"./data")
        IS_value = IS.getIS(args.outdir)
        print( filename )
        print("FID :  ",fid_value)
        print("inception score:",IS_value)
        print >> logF, "%s \nFID: %f"%(filename,fid_value)
        print >> logF, "Inception Score: "
        print >> logF, IS_value
        print >> logF,"\n\n"
    
    logF.close()
