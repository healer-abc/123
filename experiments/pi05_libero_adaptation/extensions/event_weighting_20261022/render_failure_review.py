"""CPU-only, explicitly time-indexed views of all current replay failures."""
import argparse
import json

import imageio.v2 as imageio
from PIL import Image, ImageDraw

from common import ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fine', action='store_true')
    args = parser.parse_args()
    records = json.loads((ROOT/'artifacts/replay_audit.json').read_text())['records']
    fine = {('E1-s42', 1): (0, 5), ('E2-s42', 1): (160, 5),
            ('E1-s42', 6): (130, 3), ('E1-s42', 10): (40, 3),
            ('E1-s42', 11): (25, 3)}
    for record in records:
        if record['success']:
            continue
        key = (record['run_id'], record['pair_index'])
        start, stride = fine[key] if args.fine else (0, 20 if record['suite']=='libero_10' else 5)
        reader = imageio.get_reader(record['video'])
        sheet = Image.new('RGB', (8*256, 3*288+32), 'white')
        draw = ImageDraw.Draw(sheet)
        draw.text((4, 4), f"{key} {record['suite']} task{record['task_id']} state{record['init_state_index']} events={record['events']}", fill='black')
        for i in range(24):
            step = min(start+i*stride, record['steps']-1)
            x, y = i%8*256, i//8*288+32
            draw.text((x+3, y+3), f'policy step {step}', fill='black')
            sheet.paste(Image.fromarray(reader.get_data(step)).resize((256,256)), (x,y+24))
        reader.close()
        sheet.save(ROOT/f"artifacts/failure_{record['run_id']}_pair{record['pair_index']:02d}_{'fine' if args.fine else 'coarse'}.jpg")


if __name__ == '__main__':
    main()
