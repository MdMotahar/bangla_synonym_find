import os
import re
import subprocess
import traceback
import time
import shutil
import argparse
import glob
import json
from dataProcessor_used import processData

FILEDIR = os.path.dirname(__file__)

def _translate(modelName, inputFile, outputFile,tgt_seq_length=200,eval_batch_size=1):
    cmd = f'''
        onmt_translate \
            -model \"{modelName}\" \
            -src \"{inputFile}\" \
            -output \"{outputFile}\" \
            -replace_unk copy -verbose -max_length {tgt_seq_length} -batch_size {eval_batch_size} -gpu 0
    '''
    os.system(cmd)

def translate(model_path, dataset_category, output_dir):
    src_lines, src_map = [], {}
    src_file = os.path.join(output_dir, "Outputs", f'src-{dataset_category}.txt.tok')
    with open(src_file) as f:
        lines = f.readlines()
        src_map[src_file] = len(lines)
        src_lines.extend(lines)

    merged_src_file = os.path.join(output_dir, "temp", "merged.src")
    merged_tgt_file = os.path.join(output_dir, "temp", "merged.tgt")

    with open(merged_src_file, 'w') as f:
        for line in src_lines:
            print(line.strip(), file=f)
    
    _translate(model_path, merged_src_file, merged_tgt_file)

    with open(merged_tgt_file) as inpf:
        idx = 0
        lines = inpf.readlines()

        for src_file in src_map:
            pred_file = f"pred-{dataset_category}.txt.tok".join(
                src_file.rsplit(
                    f"src-{dataset_category}.txt.tok", 1
                )
            )

            with open(pred_file, 'w') as outf:
                for _ in range(src_map[src_file]):
                    print(lines[idx].strip(), file=outf)
                    idx += 1

    # os.remove(merged_src_file)
    # os.remove(merged_tgt_file)

def calculate_scores(output_dir, dataset_category='test'):
    scores = []
    for pred_file in glob.glob(os.path.join(output_dir, "Outputs", f'*pred-{dataset_category}.txt.detok')):
        dataset_name = os.path.basename(pred_file).rsplit(
            f".pred-{dataset_category}.txt.detok", 1
        )[0]

        tgt_file_prefix = f".tgt-{dataset_category}.txt.*detok".join(
            pred_file.rsplit(
                f".pred-{dataset_category}.txt.detok", 1
            )
        )
        tgt_files = glob.glob(tgt_file_prefix)
        if tgt_files:
            bleu_cmd = [
                f"perl \"{os.path.join(FILEDIR, 'multi-bleu-detok.perl')}\"",
                f"-lc {' '.join(tgt_files)} < \"{pred_file}\""
            ]
            sacre_cmd = [
                f"cat \"{pred_file}\"",
                "|",
                f"sacrebleu {' '.join(tgt_files)}"
            ]
            
            try:
                bleu_output = str(subprocess.check_output(" ".join(bleu_cmd), shell=True)).strip()
                bleu_score = bleu_output.splitlines()[-1].split(",")[0].split("=")[1]
            except:
                bleu_score = -1

            try:
                sacre_output = str(subprocess.check_output(" ".join(sacre_cmd), shell=True)).strip()
                sacre_score = sacre_output.splitlines()[-1].split("=")[1].split()[0]
            except:
                sacre_score = -1

            scores.append(
                {
                    "dataset": dataset_name,
                    "bleu": bleu_score,
                    "sacrebleu": sacre_score
                }
            )

    return scores

def write_scores(scores, output_path):
    with open(output_path, 'w') as f:
        for model_name in scores:
            print(model_name, ":", file=f)
            for dataset_score in scores[model_name]:
                print(
                    "",
                    f"Dataset: {dataset_score['dataset']},",
                    f"BLEU: {dataset_score['bleu']},",
                    f"SACREBLEU: {dataset_score['sacrebleu']},",
                    sep="\t",
                    file=f
                )

def evaluate(eval_model,input_dir,output_dir):
    model_scores = {}
    translate(eval_model, "test", output_dir)
    processData(input_dir,output_dir, tokenization=False)
    # scores = calculate_scores(output_dir)
    # model_scores[os.path.basename(eval_model)] = scores

    # write_scores(
    #     model_scores,
    #     os.path.join(
    #         output_dir, "Reports", f"{os.path.basename(eval_model)}.test.bn2en.log"
    #     )
    # )



def translate_file(input_dir,output_dir,eval_model,):
    processData(input_dir,output_dir)
    evaluate(eval_model,input_dir,output_dir)
    
def main(args):
    translate_file(args.input_dir,args.output_dir,args.eval_model)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_dir', '-i', type=str,
        required=True,
        metavar='PATH',
        help="Input directory")

    parser.add_argument(
        '--output_dir', '-o', type=str,
        required=True,
        metavar='PATH',
        help="Output directory")

    
    parser.add_argument(
        '--src_seq_length', type=int, default=200, 
        help='maximum source sequence length')

    parser.add_argument(
        '--tgt_seq_length', type=int, default=200, 
        help='maximum target sequence length')

    parser.add_argument(
        '--eval_model', type=str, metavar="PATH", 
        help='Path to the specific model to evaluate')

    args = parser.parse_args()
    main(args)
    

    
