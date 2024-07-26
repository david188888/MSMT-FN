import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.data_loader import data_loader




train_loader= data_loader()

print(len(train_loader))

for i , batch in enumerate(train_loader):
    print(batch)
    break
