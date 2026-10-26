"""Read the compressed repository atlas without extracting it; no writes or execution."""
import argparse
import csv
import gzip
from pathlib import Path


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('prefix',nargs='?',default='experiments/pi05_libero_adaptation/')
    parser.add_argument('--catalog',choices=('project','dependencies','adjacent','directories'),default='project')
    parser.add_argument('--limit',type=int,default=40,help='0 prints all matches')
    parser.add_argument('--contains',default='')
    args=parser.parse_args()
    name='directories.csv.gz' if args.catalog=='directories' else args.catalog+'_files.csv.gz'
    root=Path(__file__).resolve().parent
    count=0
    with gzip.open(root/name,'rt',encoding='utf-8',newline='') as file:
        for row in csv.DictReader(file):
            if not row['path'].startswith(args.prefix) or args.contains not in row['path']:continue
            count+=1
            print(row['path'])
            for field in ('scope','origin','evidence','purpose','review','created_by','bytes','recursive_files','logical_bytes'):
                if field in row:print('  '+field+': '+row[field])
            if args.limit and count>=args.limit:break
    print(f'displayed {count}; use --limit 0 for all matching rows')


if __name__=='__main__':main()
