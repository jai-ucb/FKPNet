from Aug_Image import Aug_Image
import pandas as pd
import time
import glob
import multiprocessing

# ==============================================================
# 
# ==============================================================

def start():
    
	# ==============================================================
	# 
	# ==============================================================

    image_date_file = "training.csv"
    jobs_num = multiprocessing.cpu_count()
    num_images = 100 #len(train_data)
        
    train_data = pd.read_csv(image_date_file)
    train_data = train_data.dropna()
    
    images_per_job = int(num_images/jobs_num)
    image_file_name = image_date_file

    batches = []
    
    # ==============================================================
	# 
	# ==============================================================

    for i in range(0,jobs_num,1):

        image_start = images_per_job * i
        image_end = image_start + images_per_job
        vals = [image_date_file,i,images_per_job,image_start,image_end]                

        p_go = Aug_Image()
        p_go.set_config(vals)
        batches.append(p_go)
        p_go.start()
        
    # ==============================================================
	# 
	# ==============================================================

    while 1==1:
        time.sleep(10)
        for i in range(0,jobs_num,1):
            done = 0
            if "started" in str(batches[i]):
                done = 1
        if done == 0:
            break
    
    # ==============================================================
	# 
	# ==============================================================

    extension = 'csv'
    all_filenames = [i for i in glob.glob('train_data_aug*.{}'.format(extension))]
    print(all_filenames)
    combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames ])
    combined_csv.to_csv( "train_aug.csv", index=False)
    
# ==============================================================
#
# ==============================================================
if __name__ == "__main__":
	start()
# ==============================================================
#
# ==============================================================