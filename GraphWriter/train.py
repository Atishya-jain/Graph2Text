import sys
from random import shuffle
import os
from math import exp
import torch
from torch import nn
from torch.nn import functional as F
from lastDataset import dataset
from pargs import pargs,dynArgs
from models.newmodel import model

def update_lr(optimiser,args,epoch):
  if epoch%args.lrstep == 0:
    optimiser.param_groups[0]['lr'] = args.lrhigh
  else:
    optimiser.param_groups[0]['lr'] -= args.lrchange
  
  
def train(model,optimiser,dataset,args):
  print("Training",end="\t")
  loss = 0
  ex = 0
  trainorder = [('1',dataset.t1_iter),('2',dataset.t2_iter),('3',dataset.t3_iter)]
  #trainorder = reversed(trainorder)
  shuffle(trainorder)
  for spl, train_iter in trainorder:
    print(spl)
    for count, batch in enumerate(train_iter):
      #print(count)
      if count%100==99:
        print(ex,"of like 40k -- current avg loss ",(loss/ex))
      batch = dataset.fixBatch(batch)
      try:
        # returns probability of generate words concatenated with copy words, probability of copy
        preds, z, _ = model(batch)
      except Exception as e:
        print(e)
        import pdb; pdb.set_trace()
      preds = preds[:,:-1,:].contiguous()

      tgt = batch.tgt[:,1:].contiguous().view(-1).to(args.device)
      l = F.nll_loss(preds.contiguous().view(-1, preds.size(2)), tgt, ignore_index=1)
      #copy coverage (each elt at least once)
      if args.cl:
        z = z.max(1)[0]
        cl = nn.functional.mse_loss(z,torch.ones_like(z))
        l = l + args.cl*cl
        
      l.backward()
      nn.utils.clip_grad_norm_(model.parameters(),args.clip)
      loss += l.item() * len(batch.tgt)
      optimiser.step()
      optimiser.zero_grad()
      ex += len(batch.tgt)
  loss = loss/ex 
  print("AVG TRAIN LOSS: ",loss,end="\t")
  if loss < 100: print(" PPL: ",exp(loss))

def evaluate(model,dataset,args):
  print("Evaluating",end="\t")
  model.eval()
  loss = 0
  ex = 0
  for batch in dataset.val_iter:
    batch = dataset.fixBatch(batch)
    p, z, _ = model(batch)
    p = p[:,:-1,:]
    tgt = batch.tgt[:,1:].contiguous().view(-1).to(args.device)
    l = F.nll_loss(p.contiguous().view(-1,p.size(2)),tgt,ignore_index=1)
    if ex == 0:
      g = p[0].max(1)[1]
      print(dataset.reverse(g,batch.rawent[0]))
    loss += l.item() * len(batch.tgt)
    ex += len(batch.tgt)
  loss = loss/ex
  print("VAL LOSS: ",loss,end="\t")
  if loss < 100: print(" PPL: ",exp(loss))
  model.train()
  return loss

def main(args):
  try:
    os.stat(args.save)
    input("Save File Exists, OverWrite? <CTL-C> for no")
  except:
    os.mkdir(args.save)
  dset = dataset(args)
  args = dynArgs(args, dset)
  model_instance = model(args)
  print(args.device)
  model_instance = model_instance.to(args.device)
  
  if args.ckpt:
    cpt = torch.load(args.ckpt)
    model_instance.load_state_dict(cpt)
    starte = int(args.ckpt.split("/")[-1].split(".")[0])+1
    args.lr = float(args.ckpt.split("-")[-1])
    print('ckpt restored')
  else:
    with open(args.save+"/commandLineArgs.txt",'w') as f:
      f.write("\n".join(sys.argv[1:]))
    starte=0
  optimiser = torch.optim.SGD(model_instance.parameters(),lr=args.lr, momentum=0.9)

  # early stopping based on Val Loss
  lastloss = 1000000
  
  for e in range(starte,args.epochs):
    print("epoch ",e,"lr",optimiser.param_groups[0]['lr'])
    train(model_instance,optimiser,dset,args)
    vloss = evaluate(model_instance,dset,args)
    if args.lrwarm:
      update_lr(optimiser,args,e)
    print("Saving model")
    torch.save(model_instance.state_dict(),args.save+"/"+str(e)+".vloss-"+str(vloss)[:8]+".lr-"+str(optimiser.param_groups[0]['lr']))
    if vloss > lastloss:
      if args.lrdecay:
        print("decay lr")
        optimiser.param_groups[0]['lr'] *= 0.5
    lastloss = vloss
        

if __name__=="__main__":
  args = pargs()
  main(args)
