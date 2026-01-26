from training.dataset import CellDatasetRLE
from utils.visualize import show_sample

dataset = CellDatasetRLE("C:/___python/Bio-tech Project/Automated Cell & Tissue Analysis Platform/Data/images/test/image","C:/___python/Bio-tech Project/Automated Cell & Tissue Analysis Platform/Data/images/test/test.csv")
image,mask = dataset[0]
show_sample(image,mask,title="Sample Cell Image")