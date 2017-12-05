## Submitting to condor

### First time setup
```bash
git clone https://github.com/aminnj/ProjectMetis/
```

### Get the environment for submission and get code ready
```
cd ProjectMetis
source setup.sh
cd ..
cp ../gan.py .          # copy over the code to this folder
cp ../physicsfuncs.py . #
```

### Fun part

#### Generating scan list
Batch submission starts by generating a list of command line
options for `gan.py`. This is done through the `permute.py`
script. Edit in any new parameters at the top, along with a 
list of available options to scan over. Specify `NMAX` which
is the max number of random permutations to spit out. Next,
take note that some combinations of options don't make sense,
so ignore them where we say `ignore some combinations that don't
make sense`. This script prints out command line flags to stdout
so invoke with `python permute.py > allargs.txt`. Alternatively,
you can just insert arguments here by hand.

#### Submitting scan list
Submission happens in `condor.py`. At the top, specify a 
tag for this batch of submissions. Specify the hadoop output directory
and specify the input file, which should live on hadoop.
If you didn't name your scan list `allargs.txt`, then change it
in this file.

Do *not* submit to UAF (default is `T2_US_UCSD`) since we are
actually using two cores per submission slot.

Run this script as `python condor.py`.
If all goes well, you'll see one submission per line in
your scan list. Running it again will ignore scans that exist
on hadoop (done), or scans that are on condor (running)

#### Analyzing outputs
Relevant script is `check.py`. Specify the pattern for your
output directories at the top, and this script will loop through,
opening up the `history.pkl` pickle file which contains information
that is stored every 250 epochs (default). A single pickle file
with all the available information from looping over the output
directories get spit out as `total_info.pkl`. This can be analyzed
separately, by looking at the various lists inside for different models.
Keys can of course be inspected by `print blah.keys()`, but noteworthy
ones include 
* `mass_mu` - mean of invariant mass
* `mass_sig` - width of invariant mass
* `epoch` - epoch number
* `args` - dictionary of arguments/configuration/flags that this model used


