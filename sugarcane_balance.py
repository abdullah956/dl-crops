import os
import random
from glob import glob

def balance_dataset(directory):
    class_dirs = [os.path.join(directory, d) for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
    
    class_counts = {cls: len(glob(os.path.join(cls, '*'))) for cls in class_dirs}
    min_count = min(class_counts.values())
    
    for cls, count in class_counts.items():
        if count > min_count:
            images = glob(os.path.join(cls, '*'))
            random.shuffle(images)
            to_delete = images[min_count:]
            
            for img in to_delete:
                os.remove(img)
            print(f"Deleted {len(to_delete)} images from {cls}")

    print("Dataset balanced successfully!")

# Run the script on your dataset folder
balance_dataset('sugar_cane')
