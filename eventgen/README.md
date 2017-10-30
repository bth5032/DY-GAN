To generate events, run 

```BASH
root -l -b -q dyevents.C > output_filename.txt
```

You can change the output from muons to electrons in the script by uncommenting the correct pdgId. Then you want to run

```BASH
python dyev.py > line_seperated_events.input
```

This should parse the output of the Scan command and make nice line seperated inputs for training the NN. 
