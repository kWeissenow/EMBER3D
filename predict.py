import os
import time
from Bio import SeqIO
from argparse import ArgumentParser
from Ember3D import *

parser = ArgumentParser()
parser.add_argument('-i', '--input', dest="input", type=str, required=True)
parser.add_argument('-o', '--output_dir', dest='output_dir', type=str, required=True)
parser.add_argument('-d', '--device', default='cuda:0', dest="device", type=str)
parser.add_argument('--output-2d', dest="output_2d", action="store_true")
parser.add_argument('--no-pdb', dest="no_pdb", action="store_true")
parser.add_argument('--no-distance-map', dest="no_distance_map", action="store_true")
parser.add_argument('--save-distance-array', dest="save_distance_array", action="store_true")
parser.add_argument('-m', '--model', default="model/EMBER3D.model", dest='model_checkpoint', type=str)
parser.add_argument('--t5-dir', dest='t5_dir', default="./ProtT5-XL-U50/", type=str)
parser.add_argument('--prostt5', dest="prostt5", action="store_true")
args = parser.parse_args()

use_prostt5 = False
if args.prostt5:
    use_prostt5 = True
    # Switch to Ankh model checkpoint if default argument was used
    if args.model_checkpoint == "model/EMBER3D.model":
        args.model_checkpoint = "model/EMBER3D_ProstT5.model"

# Output directories
pdb_dir = os.path.join(args.output_dir, "pdb")
image_dir = os.path.join(args.output_dir, "images")
dist_orient_dir = os.path.join(args.output_dir, "output_2d")
if not os.path.isdir(args.output_dir):
    os.makedirs(args.output_dir)
if not args.no_pdb and not os.path.isdir(pdb_dir):
    os.makedirs(pdb_dir)
if not args.no_distance_map and not os.path.isdir(image_dir):
    os.makedirs(image_dir)
if (args.output_2d or args.save_distance_array) and not os.path.isdir(dist_orient_dir):
    os.makedirs(dist_orient_dir)

# Prediction
Ember3D = Ember3D(args.model_checkpoint, args.t5_dir, args.device, use_prostt5)

start_time = time.time()
for i,record in enumerate(SeqIO.parse(args.input, "fasta")):
    id = record.id
    seq = str(record.seq).upper()
    if seq.endswith('*'):
        seq = seq[:-1]

    print("{}\t{}".format(i, id))

    aa_list = list("ACDEFGHIKLMNPQRSTVWY")
    skip = False
    for c in seq:
        if c not in aa_list:
            skip = True
            break
    if skip:
        print("Skipping {} because of unknown residues".format(id))
        continue

    try:
        with torch.cuda.amp.autocast():
            result = Ember3D.predict(seq)

            if args.output_2d:
                result.save_2d_output("{}/{}.npz".format(dist_orient_dir, id))

            if args.save_distance_array:
                dist_map = result.get_distance_map()
                np.save("{}/{}_distances.npy".format(dist_orient_dir, id), dist_map)

            if not args.no_pdb:
                result.save_pdb(id, "{}/{}.pdb".format(pdb_dir, id))

            if not args.no_distance_map:
                result.save_distance_map("{}/{}.png".format(image_dir, id))
    except RuntimeError:
        print("Out of memory! Skipping..")

end_time = time.time()
print("Total prediction time: {} seconds".format(end_time - start_time))
