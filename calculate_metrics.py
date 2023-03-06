import os
import math

log_dir = './logs'
dataset = 'nerf'
dataset_dir = os.path.join(log_dir, dataset)

scene = '3view_ft'


metrics = {'PSNR':0, 'SSIM':0, 'LPIPS':0}
num = 1e-6

for basedir in os.listdir(dataset_dir):
    if scene not in basedir:
        metric_path = os.path.join(dataset_dir, basedir, 'metrics.txt')
        
        if os.path.isfile(metric_path):
            f = open(metric_path)
        
            for k in metrics.keys():
                line = f.readline()
                if line[-1] == '\n':
                    line = line[:-1]
                val = float(line[len(k) + 1:])
                
                if not math.isnan(val):
                    metrics[k] += val
                    num += 1

for k in metrics.keys():
    metrics[k] = metrics[k] * 3 / num
    print(f'{k} is {metrics[k]}')

print(f'{int(num/3)} valid scenes')
