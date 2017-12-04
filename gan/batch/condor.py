from pprint import pprint

from metis.Utils import condor_q, get_hist, condor_submit

import math, sys, os

"""
Some simple uses of the condor_q API
"""


if __name__ == "__main__":

    submission_tag = "v3"
    outputdir = "/hadoop/cms/store/user/namin/gan_output/{}".format(submission_tag)
    input_file = "/hadoop/cms/store/user/namin/data_xyz.npy"
    scanlist = "allargs.txt"

    os.system("mkdir -p {}".format(outputdir))

    my_jobs = condor_q(selection_pairs=[["GANtag",submission_tag]],extra_columns=["job_num"])
    already_running = [int(job["job_num"]) for job in my_jobs]

    all_args = []
    with open(scanlist,"r") as fhin:
        for line in fhin:
            parts = line.strip().split("|")
            if len(parts) > 0:
                parts += [" --input_file {}".format(input_file)]
                all_args.append(parts)
    for job_num, pyargs in enumerate(all_args):
        if job_num in already_running: continue # don't resubmit if already running
        job_tag = "{}_{}".format(submission_tag,job_num)
        if os.path.exists("{}/{}/history.pkl".format(outputdir, job_tag)): continue # don't resubmit if already finished
        os.system("mkdir -p {}/{}".format(outputdir, job_tag))
        arguments = [ outputdir, job_tag ] + pyargs
        params = {
                "executable": "executable.sh",
                "arguments": arguments,
                "inputfiles": ["gan.py","physicsfuncs.py"],
                "logdir": "logs/",
                "selection_pairs": [["GANtag",submission_tag],["job_num",job_num]],
                # "sites": "T2_US_UCSD,UAF",
                "sites": "T2_US_UCSD",
                # "sites": "UAF",
                }
        succeeded, cid = condor_submit(**params)
        if succeeded:
            print "Submitted job {} to {}.".format(job_num, cid)
        else:
            print "ERROR with job {}".format(job_num)

        # print "FIXME"
        # break

#     all_statuses = [job["JobStatus"] for job in my_jobs]
#     print "______ STATUS ______"
#     print get_hist(all_statuses)
