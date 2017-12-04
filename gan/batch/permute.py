from itertools import izip, product
import random

NMAX = 800

params = {
"batch_size": [100,200,500,1000],
"noise_type": [1,2,3],
"noise_size": [8,16,32],
"do_concatenate_disc": [True,False],
"do_concatenate_gen": [True,False],
"do_batch_normalization_disc": [True,False],
"do_batch_normalization_gen": [True,False],
"do_soft_labels": [True,False],
"do_noisy_labels": [True,False],
"nepochs_decay_noisy_labels": [500,1000,5000,10000],
"use_ptetaphi_additionally": [True,False],
"scaler_type": ["minmax", "robust", "standard", "none"],
"do_tanh_gen": [True,False],
"optimizer_disc": ["adadelta","adam"],
"optimizer_gen": ["adadelta","adam"],
}

def my_product(dicts):
    return (dict(izip(dicts, x)) for x in product(*dicts.values()))

products = list(my_product(params))
random.shuffle(products)

all_args = []
for product in products[:NMAX]:
    # ignore some combinations that don't make sense
    if product["noise_type"] == 3 and product["noise_size"] == 8: continue
    if product["do_tanh_gen"] and product["scaler_type"] == "none": continue
    args = []
    for k,v in product.items():
        if type(v) == bool and v == False: continue
        if type(v) == bool and v == True:
            args.append(" --{}".format(k,v))
        else:
            args.append(" --{} {}".format(k,v))
    all_args.append(args)
# print len(all_args)
# print all_args

for args in all_args:
    print "|".join(args)

