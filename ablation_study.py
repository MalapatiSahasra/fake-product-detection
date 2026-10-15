import glob, random, csv, os, json, time
from modules import logo_matcher, color_inspector, typo_detective

def load_manifest(path):
    m = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            brand = row['source_seed_image'].split('_')[0]
            m[row['output_filename']] = brand
    return m

gen_manifest = load_manifest('dataset/genuine/_manifest.csv')
cf_manifest = load_manifest('dataset/counterfeit/_manifest.csv')

gallery_paths = sorted(glob.glob('dataset/gallery/*.jpg'))
sample_gallery = random.Random(42).sample(gallery_paths, min(20, len(gallery_paths)))
logo_g = logo_matcher.build_gallery(sample_gallery)
color_g = color_inspector.build_gallery(sample_gallery)
brands_list = ['NIKE', 'PUMA', 'ADIDAS', 'GUCCI', 'LV']

genuine_images = sorted(glob.glob('dataset/genuine/*.jpg'))
counterfeit_images = sorted(glob.glob('dataset/counterfeit/*.jpg'))

records = []
start = time.time()
total = len(genuine_images) + len(counterfeit_images)
done = 0
for path, truth in [(p, 'genuine') for p in genuine_images] + [(p, 'counterfeit') for p in counterfeit_images]:
    manifest = gen_manifest if truth == 'genuine' else cf_manifest
    brand = manifest.get(os.path.basename(path), 'unknown')
    try:
        r_logo = logo_matcher.verify_gallery(logo_g, path)
        r_color = color_inspector.verify_gallery(color_g, path)
        r_typo = typo_detective.verify_gallery(brands_list, path)
    except Exception as e:
        done += 1
        continue
    score = sum([r_logo['passed'], r_color['passed'], r_typo['passed']])
    cascade_pass = score >= 2
    records.append({
        'brand': brand, 'truth': truth,
        'logo': r_logo['passed'], 'color': r_color['passed'], 'typo': r_typo['passed'],
        'cascade': cascade_pass,
    })
    done += 1
    if done % 100 == 0:
        print(f'{done}/{total} elapsed={time.time()-start:.0f}s', flush=True)

with open('/tmp/ablation_records.json', 'w') as f:
    json.dump(records, f)
print('DONE', len(records), 'records', time.time()-start, 's')
