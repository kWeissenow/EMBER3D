from pathlib import Path
import sys
import os
import numpy as np
import shutil
import matplotlib.pyplot as plt
from Bio import SeqIO
from PIL import Image, ImageFont, ImageDraw
from progressBar import *

if "pymol" not in  "\t".join(sys.path):
    sys.path.append("/usr/bin/pymol")
    sys.path.append("/usr/lib/python")
    import __main__
    __main__.pymol_argv = ['pymol','-Qqc']
    import pymol
    pymol.finish_launching()
else: # This IF avoids re-appending to PATH in case you execute this snippet multiple times in the same session
    import pymol

pymol.cmd.feedback("disable","all","actions")
pymol.cmd.feedback("disable","all","results")

def write_png(id, pdb_file, png_file, width=720, height=720):
    pdb_name = pdb_file.name.split('.')[0]
    parts = pdb_name.split("_")
    suffix = parts[-1]
    from_res = suffix[0]
    to_res = suffix[-1]
    mid = suffix[1:-1]
    try:
        residx = int(mid)
    except ValueError:
        residx = 0

    pymol.cmd.load(pdb_file, pdb_name)
    pymol.cmd.disable("all")
    pymol.cmd.enable(pdb_name)
    pymol.cmd.hide('all')
    pymol.cmd.show('cartoon')
    pymol.cmd.set('ray_opaque_background', 1)
    pymol.cmd.bg_color('white')
    pymol.cmd.spectrum('b', palette='red red red orange yellow cyan blue', minimum=0, maximum=100)
    pymol.cmd.color('nitrogen', 'resi {}'.format(residx))
    pymol.cmd.png(str(png_file), width=width, height=height, quiet=1)
    pymol.cmd.delete(pdb_name)

    img = Image.open(png_file)
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("font.ttf", 16)
    draw.text((1, 1), suffix, (0,0,0), font=font)
    img.save(png_file)

def main():
    input_fasta = sys.argv[1]
    prediction_path = sys.argv[2]
    
    movie_width = 1280
    movie_height = 720
    
    if len(sys.argv) > 3:
        movie_width = int(sys.argv[3])
    if len(sys.argv) > 4:
        movie_height = int(sys.argv[4])
    
    movie_right_panel_x = int(movie_width * 0.6)
    movie_right_panel_width = movie_width - movie_right_panel_x

    aa_list = list("ACDEFGHIKLMNPQRSTVWY")
    for record in SeqIO.parse(input_fasta, "fasta"):
        id = record.id
        seq = list(record.seq)

        pdb_dir = Path("{}/{}/pdb".format(prediction_path, id))
        png_dir = Path("{}/{}/png".format(prediction_path, id))
        out_dir = Path("{}/{}/out".format(prediction_path, id))
        dist_dir = Path("{}/{}/distance_arrays".format(prediction_path, id))
        dist_img_dir = Path("{}/{}/distance_images".format(prediction_path, id))
        mut_matrix_file = Path("{}/{}/{}_mutation_matrix.png".format(prediction_path, id, id))
        if not os.path.isdir(png_dir):
            os.makedirs(png_dir)
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)
        if not os.path.isdir(dist_img_dir):
            os.makedirs(dist_img_dir)

        mut_matrix_img = Image.open(mut_matrix_file)
        scale_factor = movie_right_panel_width / mut_matrix_img.width
        mut_matrix_img = mut_matrix_img.resize((int(mut_matrix_img.width * scale_factor), int(mut_matrix_img.height * scale_factor)))

        print("Rendering frames for {}".format(id))
        total = len(seq) * 19
        printProgressBar(0, total, prefix = 'Frames:', suffix = 'Complete', length = 50)
        counter = 1
        for i in range(len(seq)):
            for aa in aa_list:
                if aa == seq[i]:
                    continue
                temp_seq = seq.copy()
                temp_seq[i] = aa

                pdb_file = pdb_dir / "{}_{}{}{}.pdb".format(id, seq[i], i+1, aa)
                png_file = png_dir / "{}.png".format(counter)
                dist_file = dist_dir / "{}_{}{}{}_distances.png".format(id, seq[i], i+1, aa)
                dist_array_file = dist_dir / "{}_{}{}{}_distances.npy".format(id, seq[i], i+1, aa)
                dist_img_file = dist_img_dir / "{}_{}{}{}.png".format(id, seq[i], i+1, aa)
                out_file = out_dir / "{}.png".format(counter)

                write_png(id, pdb_file, png_file, width=movie_right_panel_x, height=movie_height)
                
                dist_map = np.load(dist_array_file)
                plt.imshow(dist_map, cmap='hot_r')
                clb = plt.colorbar()
                clb.ax.set_ylabel("Angstrom distance", rotation=90)
                plt.savefig(dist_img_file, bbox_inches='tight')
                plt.close()
                
                img_3d = Image.open(png_file)
                dist_2d = Image.open(dist_img_file)
                scale_factor = movie_right_panel_width / dist_2d.width
                dist_2d = dist_2d.resize((int(dist_2d.width * scale_factor), int(dist_2d.height * scale_factor)))
                
                tar_img = Image.new('RGB', (movie_width, movie_height), (255,255,255))
                tar_img.paste(img_3d, (0,0))
                tar_img.paste(dist_2d, (movie_right_panel_x,0))
                tar_img.paste(mut_matrix_img, (movie_right_panel_x, dist_2d.height))
                tar_img.save(out_file)

                printProgressBar(counter, total, prefix = 'Frames:', suffix = 'Complete', length = 50)
                counter += 1

        print("")
        print("Rendering movie for {}".format(id))
        os.system("ffmpeg -y -f image2 -framerate 19 -i {}/%d.png -vcodec libx264 -crf 25 -pix_fmt yuv420p {}/{}/{}.mp4 > /dev/null 2>&1".format(out_dir, prediction_path, id, id))

        # Cleanup
        print("Done, cleaning up..")
        shutil.rmtree(png_dir)
        shutil.rmtree(out_dir)
        print("")

if __name__ == "__main__":
    main()

