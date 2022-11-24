import numpy as np
import os
import matplotlib.pyplot as plt
from progressBar import *
from Bio import SeqIO
from argparse import ArgumentParser
from Ember3D import *
from lddt import *


parser = ArgumentParser()
parser.add_argument('-i', '--input', dest="input", type=str, required=True)
parser.add_argument('-o', '--output_dir', dest='output_dir', type=str, required=True)
parser.add_argument('-d', '--device', default='cuda:0', dest="device", type=str)
parser.add_argument('--no-pdb', dest="no_pdb", action="store_true")
parser.add_argument('--no-distance-map', dest="no_distance_map", action="store_true")
parser.add_argument('--save-distance-array', dest="save_distance_array", action="store_true")
parser.add_argument('-m', '--model', default="model/EMBER3D.model", dest='model_checkpoint', type=str)
parser.add_argument('--t5-dir', dest='t5_dir', default="./ProtT5-XL-U50/", type=str)
args = parser.parse_args()

# Output directory
if not os.path.isdir(args.output_dir):
    os.makedirs(args.output_dir)

Ember3D = Ember3D(args.model_checkpoint, args.t5_dir, args.device)

for i,record in enumerate(SeqIO.parse(args.input, "fasta")):
    id = record.id
    seq = str(record.seq)
    seq_l = list(seq)
    length = len(seq)

    if "X" in seq:
        print("Skipping {} because of unknown residues".format(id))
        continue

    sample_dir = os.path.join(args.output_dir, id)
    if not os.path.isdir(sample_dir):
        os.makedirs(sample_dir)
    pdb_dir = os.path.join(sample_dir, "pdb")
    image_dir = os.path.join(sample_dir, "images")
    dist_dir = os.path.join(sample_dir, "distance_arrays")
    if not args.no_pdb and not os.path.isdir(pdb_dir):
        os.makedirs(pdb_dir)
    if not args.no_distance_map and not os.path.isdir(image_dir):
        os.makedirs(image_dir)
    if args.save_distance_array and not os.path.isdir(dist_dir):
        os.makedirs(dist_dir)

    # Wild-type prediction

    print("{}\t{}".format(i, id))
    with torch.cuda.amp.autocast():
        result = Ember3D.predict(seq)

        if not args.no_pdb:
            result.save_pdb(id, "{}/{}.pdb".format(pdb_dir, id))

        if not args.no_distance_map:
            result.save_distance_map("{}/{}.png".format(image_dir, id))

        wild_type_distance_map = result.get_distance_map()

        if args.save_distance_array:
            np.save("{}/{}_distances.npy".format(dist_dir, id), wild_type_distance_map)


    # Mutation predictions

    aa_list = list("ACDEFGHIKLMNPQRSTVWY")
    mutation_matrix = np.ones((20, length))
    mutation_log = os.path.join(sample_dir, id + "_mutation_log.txt")
    total = length * 19
    counter = 0
    printProgressBar(0, total, prefix = 'Mutants:', suffix = 'Complete', length = 50)
    with open(mutation_log, "w") as f:
        for seq_idx in range(length):
            for aa in aa_list:
                if aa == seq_l[seq_idx]:
                    continue
                mut_seq = seq_l.copy()
                mut_seq[seq_idx] = aa
                mut_seq = "".join(mut_seq)
                mut_id = "{}_{}{}{}".format(id, seq_l[seq_idx], seq_idx + 1, aa)

                with torch.cuda.amp.autocast():
                    result = Ember3D.predict(mut_seq)

                    if not args.no_pdb:
                        result.save_pdb(mut_id, "{}/{}.pdb".format(pdb_dir, mut_id))

                    if not args.no_distance_map:
                        result.save_distance_map("{}/{}.png".format(image_dir, mut_id))

                    mutant_distance_map = result.get_distance_map()

                    if args.save_distance_array:
                        np.save("{}/{}_distances.npy".format(dist_dir, mut_id), mutant_distance_map)

                    struct_diff = lddt(torch.from_numpy(mutant_distance_map), torch.from_numpy(wild_type_distance_map))
                    struct_diff = torch.mean(struct_diff).item()
                    mutation_matrix[aa_list.index(aa), seq_idx] = struct_diff
                    f.write("{} {} {} {}\n".format(seq_l[seq_idx], seq_idx + 1, aa, struct_diff))

                    counter += 1
                    printProgressBar(counter, total, prefix = 'Mutants:', suffix = 'Complete', length = 50)

    print("")

    plt.rc('ytick', labelsize=6)
    rg = np.arange(1, length+1)
    nx = rg.shape[0]
    x_positions = np.arange(0, nx+1, 10)
    x_positions -= 1
    x_positions[0] += 1
    x_labels = rg[x_positions]
    plt.xticks(x_positions, x_labels)
    plt.yticks(ticks=range(20), labels=aa_list)
    plt.imshow(mutation_matrix, cmap="gist_heat", vmin=0.0, vmax=1.0)
    cbar = plt.colorbar(shrink=0.5)
    cbar.set_label("similarity to wild-type")
    plt.title("Effect of point mutations")

#    plt.gca().set_axis_off()
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    plt.margins(0,0)
#    plt.gca().xaxis.set_major_locator(plt.NullLocator())
#    plt.gca().yaxis.set_major_locator(plt.NullLocator())

    plt.savefig("{}/{}_mutation_matrix.png".format(sample_dir, id), bbox_inches = 'tight', pad_inches = 0)
    plt.clf()
