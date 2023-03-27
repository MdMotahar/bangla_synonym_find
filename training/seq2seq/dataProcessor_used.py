import numpy as np
import argparse
import sys
import shutil
import pyonmttok
import os
import glob
import math
from tqdm import tqdm

def createFolders(output_dir):
    required_dirnames = [
        "data",
        "Outputs",
        "temp",
        "Preprocessed",
        "Reports",
        "Models"
    ]

    # do cleanup first
    for dirname in required_dirnames[:3]:
        if os.path.isdir(os.path.join(output_dir, dirname)):
            shutil.rmtree(os.path.join(output_dir, dirname))

    
    if os.path.isdir(os.path.join(output_dir, "Preprocessed")):
        shutil.rmtree(os.path.join(output_dir, "Preprocessed"))

    for dirname in required_dirnames:
        os.makedirs(os.path.join(output_dir, dirname), exist_ok=True)


def _move(input_dir,output_dir,src_lang, dataset_category='test'):
    src_file = os.path.join(input_dir, "data", f"{dataset_category}.{src_lang}")
    # tgt_file = os.path.join(input_dir, "data", f"{dataset_category}.{tgt_lang}")
    # print(src_file,tgt_file)
    shutil.copy(
                src_file,
                os.path.join(
                    output_dir,
                    "Outputs",
                    f"src-{dataset_category}.txt"
                ) 
            )
    # shutil.copy(
    #             tgt_file, 
    #             os.path.join(
    #                 output_dir,
    #                 "Outputs",
    #                 f"tgt-{dataset_category}.txt"
    #             )
    #         )
        
def moveRawData(input_dir,output_dir,src_lang,tgt_lang):
    # move vocab models
    shutil.copy(
        os.path.join(input_dir, "vocab", f"{src_lang}.model"),
        os.path.join(output_dir, "Preprocessed", "srcSPM.model")
    )
    shutil.copy(
        os.path.join(input_dir, "vocab", f"{tgt_lang}.model"),
        os.path.join(output_dir, "Preprocessed", "tgtSPM.model")
    )

    vocab_cmd = [
        "spm_export_vocab --model",
        os.path.join(output_dir, "Preprocessed", "srcSPM.model"),
        "| tail -n +4 >",
        os.path.join(output_dir, "Preprocessed", "srcSPM.vocab")
    ]
    os.system(" ".join(vocab_cmd))

    vocab_cmd = [
        "spm_export_vocab --model",
        os.path.join(output_dir, "Preprocessed", "tgtSPM.model"),
        "| tail -n +4 >",
        os.path.join(output_dir, "Preprocessed", "tgtSPM.vocab")
    ]
    os.system(" ".join(vocab_cmd))

    _move(input_dir,output_dir,src_lang)
        

def spmOperate(output_dir, fileType, tokenize):
    if tokenize:
        modelName = os.path.join(output_dir, "Preprocessed", f"{fileType}SPM.model")
        input_files = glob.glob(os.path.join(output_dir, "Outputs", f'*{fileType}-*'))
        
        for input_file in input_files:
            
            spm_cmd = [
                f"spm_encode --model=\"{modelName}\"",
                f"--output_format=piece",
                f"< \"{input_file}\" > \"{input_file}.tok\""
            ]
            os.system(" ".join(spm_cmd))
            os.remove(input_file)

    else:
        modelName = os.path.join(output_dir, "Preprocessed", f"tgtSPM.model")
        for input_file in glob.glob(os.path.join(output_dir, "Outputs", f'*{fileType}-*.tok')):
            
            spm_cmd = [
                f"spm_decode --model=\"{modelName}\"",
                f"< \"{input_file}\" > \"{'.detok'.join(input_file.rsplit('.tok', 1))}\""
            ]
            os.system(" ".join(spm_cmd))
            os.remove(input_file)
            post_cmd = f"""sed 's/▁/ /g;s/  */ /g' -i \"{'.detok'.join(input_file.rsplit('.tok', 1))}\""""
            os.system(post_cmd)
        
        
def tokenize(output_dir):
    spmOperate(output_dir, 'src', tokenize=True)
    # spmOperate(output_dir, 'tgt', tokenize=True)
            
def detokenize(output_dir):        
    # spmOperate(output_dir, 'tgt', tokenize=False)
    spmOperate(output_dir, 'pred', tokenize=False)

def processData(input_dir,output_dir,src_lang='bn',tgt_lang='en', dataset_category='test',tokenization=True):
    if tokenization:
        createFolders(output_dir)
        moveRawData(input_dir,output_dir,src_lang,tgt_lang)
        tokenize(output_dir)
    else:
        detokenize(output_dir)
