'''
train
 image
 label
 edge
val
 image
 label
 edge
test
 image
'''

import sys
import os
def to_list(root,split,img,mask,edge):
  def check_exist(filename,dirname):
    if dirname is None:
      return False
    if not os.path.exists(os.path.join(root,split,dirname,filename)):
      return False
    return True
  img_names=os.listdir(os.path.join(root,split,img))
  split_lst=[]
  for name in img_names:
    _split_lst=[]
    if check_exist(name,img):
      _split_lst.append(os.path.join(split,img,name))
    if check_exist(name,mask):
      _split_lst.append(os.path.join(split,mask,name))
    if check_exist(name,edge):
      _split_lst.append(os.path.join(split,edge,name))
    split_lst.append(_split_lst)
  return split_lst

root=sys.argv[1]

lst=dict()
for split in ['train','val']:
  img_path='image'
  mask_path='label'
  edge_path='edge'
  if split=='test':
    mask_path=None
    edge_path=None

  lst[split]=to_list(root,split,img_path,mask_path,edge_path)
  print(split,len(lst[split]))

lst['trainval']=lst['train']+lst['val']

for k,v in lst.items():
  with open(k+'.lst','w') as f:
    for item in v:
      f.write((' ').join(item)+'\n')
   
