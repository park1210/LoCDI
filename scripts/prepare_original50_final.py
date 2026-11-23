"""Fresh Source audit and network-denied integration; never starts generation."""
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
NS='ORIGINAL50_FINAL_EXECUTION_20261123_v1'
PUB=ROOT/'outputs/research_evaluation'/NS
PR=ROOT/'outputs/private/research_evaluation'/NS

def main():
    q=argparse.ArgumentParser();q.add_argument('mode',choices=['audit','fake']);a=q.parse_args()
    if a.mode=='audit':
        from scripts import prepare_original50_full_readiness as legacy
        legacy.PUB=PUB;legacy.PR=PR
        legacy.audit()
    else:
        from scripts import probe_original50_full_offline as legacy
        legacy.PUB=PUB;legacy.PR=PR
        legacy.main()

if __name__=='__main__':main()
